import time
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             accuracy_score, precision_score, 
                             recall_score, fbeta_score)

class MultilabelEvaluator:
    def __init__(self, model, processor, img_val_path, txt_val_path, label_cols, threshold=0.5):
        self.model = model
        self.processor = processor
        self.label_cols = list(label_cols)
        self.threshold = threshold
        
        # Load validation datasets
        self.img_val_df = pd.read_csv(img_val_path).fillna(0)
        self.txt_val_df = pd.read_csv(txt_val_path).fillna(0)
        
        # Robustly find the text column
        self.txt_col = 'sentence'
        for col in self.txt_val_df.columns:
            if col.strip().lower() in ['sentence', 'report', 'text', 'caption']:
                self.txt_col = col
                break
                
        # Robustly find the image column
        self.img_col = 'image_path'
        for col in self.img_val_df.columns:
            if col.strip().lower() in ['image_path', 'imgpath', 'path', 'image path', 'img_path']:
                self.img_col = col
                break

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        start_inf = time.time()
        
        no_finding_idx = -1
        if 'No Finding' in self.label_cols:
            no_finding_idx = self.label_cols.index('No Finding')
            
        pathology_embeds = []
        for col in self.label_cols:
            positive_sentences = self.txt_val_df[self.txt_val_df[col] == 1][self.txt_col].tolist()
            if not positive_sentences:
                positive_sentences = [f"Patient has {col}"]
                
            inputs = self.processor(text=positive_sentences, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items() if k != 'pixel_values'}
            
            # This naturally uses the frozen BERT base + the actively learning text projection head
            text_embed = self.model.encode_text(inputs['input_ids'], inputs['attention_mask'])
            avg_embed = text_embed.mean(dim=0, keepdim=True)
            avg_embed = avg_embed / avg_embed.norm(dim=-1, keepdim=True)
            pathology_embeds.append(avg_embed)
            
        pathology_embeds = torch.cat(pathology_embeds, dim=0)

        all_preds, all_probs, all_labels = [], [], []
        
        for _, row in self.img_val_df.iterrows():
            img = Image.open(row[self.img_col]).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            
            img_embed = self.model.encode_image(inputs['pixel_values'].cuda())
            logits = self.model.compute_logits(img_embed, pathology_embeds)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            preds = (probs > self.threshold).astype(int)
            
            # Apply No Finding Override Rule
            if no_finding_idx != -1 and np.argmax(probs) == no_finding_idx:
                preds = np.zeros_like(preds)
                preds[no_finding_idx] = 1
            
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(row[self.label_cols].values.astype(int))
            
        inf_time = time.time() - start_inf
        y_true, y_pred, y_prob = np.array(all_labels), np.array(all_preds), np.array(all_probs)
        
        metrics = {}
        
        aucs, auprcs = [], []
        for i in range(y_true.shape[1]):
            if np.sum(y_true[:, i]) > 0 and np.sum(y_true[:, i]) < len(y_true):
                aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
                auprcs.append(average_precision_score(y_true[:, i], y_prob[:, i]))
                
        metrics['macro_auc'] = np.mean(aucs) if len(aucs) > 0 else 0.0
        metrics['macro_auprc'] = np.mean(auprcs) if len(auprcs) > 0 else 0.0
        
        try:
            metrics['micro_auc'] = roc_auc_score(y_true, y_prob, average='micro')
            metrics['micro_auprc'] = average_precision_score(y_true, y_prob, average='micro')
        except ValueError:
            metrics['micro_auc'], metrics['micro_auprc'] = 0.0, 0.0
            
        metrics['acc'] = accuracy_score(y_true.flatten(), y_pred.flatten())
        metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1'] = fbeta_score(y_true, y_pred, beta=1, average='macro', zero_division=0)
        metrics['f2'] = fbeta_score(y_true, y_pred, beta=2, average='macro', zero_division=0)
        metrics['f3'] = fbeta_score(y_true, y_pred, beta=3, average='macro', zero_division=0)
        
        print(f"\n--- Multilabel Evaluation (Inf Time: {inf_time:.2f}s) ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        # Re-enable training mode before exiting
        self.model.train()
        return metrics

# Legacy support
class Evaluator:
    def __init__(self, medclip_clf, eval_dataloader, mode):
        self.medclip_clf = medclip_clf
        self.eval_dataloader = eval_dataloader
        self.mode = mode
    def evaluate(self):
        return {'acc': 0.0, 'auc': 0.0}