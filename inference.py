import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T

# 1. Get the current directory of this script (JIT-Webapp)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Import MedCLIP directly since it is now in the same directory
from medclip import constants
from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip.dataset import MedCLIPProcessor

# Environment setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
constants.BERT_TYPE = 'emilyalsentzer/Bio_ClinicalBERT'
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class MedCLIPPredictor:
    def __init__(self):
        self.processor = MedCLIPProcessor()
        
        # Map the frontend plane choices to your specific dataset folders
        self.plane_to_dataset = {
            "sagittal": "SPIDER",
            "axial": "Mendeley"
        }
        
        # Cache for models and embeddings
        self.models = {}
        self.label_cols = {}
        self.pathology_embeds = {}

    def get_paths(self, plane):
        target_dataset = self.plane_to_dataset[plane]
        
        # Resolve paths based on the local JIT-Webapp structure
        weight_path = os.path.join(CURRENT_DIR, "weights", target_dataset, "pytorch_model.bin")
        drw_path = os.path.join(CURRENT_DIR, "text_val", target_dataset, "drw_weights.pt")
        txt_val_path = os.path.join(CURRENT_DIR, "text_val", target_dataset, "txt_val.csv")
        
        return weight_path, drw_path, txt_val_path

    def load_model_for_plane(self, plane):
        if plane not in self.models:
            print(f"Loading {plane} model and text embeddings into memory...")
            weight_path, drw_path, txt_val_path = self.get_paths(plane)
            
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"Model weight not found: {weight_path}")
            if not os.path.exists(txt_val_path):
                raise FileNotFoundError(f"Text validation file not found: {txt_val_path}")
            
            # 1. Load labels
            drw_payload = torch.load(drw_path, map_location='cpu', weights_only=False)
            labels = list(drw_payload['label_cols'])
            self.label_cols[plane] = labels
            
            # 2. Load Model
            model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE), strict=False)
            model.to(DEVICE)
            model.eval()
            self.models[plane] = model
            
            # 3. Load text dataset
            txt_val_df = pd.read_csv(txt_val_path).fillna(0)
            txt_col = 'sentence'
            for col in txt_val_df.columns:
                if col.strip().lower() in ['sentence', 'report', 'text', 'caption']:
                    txt_col = col
                    break
            
            # 4. Precompute Averaged Text Embeddings Class-wise
            embeds = []
            with torch.no_grad():
                for col in labels:
                    positive_sentences = txt_val_df[txt_val_df[col] == 1][txt_col].tolist()
                    
                    if not positive_sentences:
                        positive_sentences = [f"Patient has {col}"]
                        
                    inputs = self.processor(text=positive_sentences, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k != 'pixel_values'}
                    
                    text_embed = model.encode_text(inputs['input_ids'], inputs['attention_mask'])
                    avg_embed = text_embed.mean(dim=0, keepdim=True)
                    avg_embed = avg_embed / avg_embed.norm(dim=-1, keepdim=True)
                    embeds.append(avg_embed)
                    
            self.pathology_embeds[plane] = torch.cat(embeds, dim=0)
            print(f"[{plane.upper()}] Loaded model and averaged embeddings for {len(labels)} classes.")
            
        return self.models[plane], self.label_cols[plane], self.pathology_embeds[plane]

    def predict(self, image_path, plane, threshold=0.5):
        model, labels, text_embeds = self.load_model_for_plane(plane)
        
        img = Image.open(image_path).convert("RGB")
        
        # BYPASS BUGGY PROCESSOR: Manually apply the standard ViT transformations 
        eval_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform the image and add the batch dimension to create a (1, 3, 224, 224) tensor
        pixel_values = eval_transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # Feed the perfectly formatted tensor directly into the model
            img_embed = model.encode_image(pixel_values)
            logits = model.compute_logits(img_embed, text_embeds)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
        # Handle "No Finding" Override
        no_finding_idx = labels.index('No Finding') if 'No Finding' in labels else -1
        is_no_finding = False
        if no_finding_idx != -1 and np.argmax(probs) == no_finding_idx:
            is_no_finding = True

        results = []
        for i, label in enumerate(labels):
            prob = float(probs[i])
            
            if is_no_finding:
                pred_binary = 1 if i == no_finding_idx else 0
            else:
                pred_binary = 1 if prob > threshold else 0
                
            results.append({
                "label": label,
                "probability": round(prob, 4),
                "prediction": pred_binary
            })
            
        results.sort(key=lambda x: x['probability'], reverse=True)
        return results