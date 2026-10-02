import os
import json
import pdb
import time
import psutil
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
from collections import defaultdict
import math

import numpy as np
import torch
from torch import nn
from torch import device, Tensor
from tqdm.autonotebook import trange
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import distributed as dist
import transformers

WEIGHTS_NAME = "pytorch_model.bin"

class Trainer:
    '''trainer for single-gpu training.
    '''
    def __init__(self, args=None):
        pass

    def train(self,
        model,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        eval_dataloader = None,
        evaluator=None,
        epochs: int = 1,
        steps_per_epoch = None,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 100,
        save_steps : int = 100,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        accumulation_steps: int = 1,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_total_limit: int = 0,
        load_best_model_at_last: bool = True,
        drw_ratio: float = 0.8,
        use_drw: bool = True, # <-- Re-added the toggle here
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        self.best_score = -9999999
        self.accumulation_steps = accumulation_steps
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.score_logs = defaultdict(list)
        self.evaluator = evaluator
        self.eval_dataloader = eval_dataloader

        dataloaders = [dataloader for dataloader,_,_ in train_objectives]
        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        loss_models = [loss for _, loss,_ in train_objectives]
        train_weights = [weight for _,_,weight in train_objectives]

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        # map models to devices
        model = model.cuda()

        # execute training on multiple GPUs
        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        train_loss_dict = defaultdict(list)
        
        train_start_time = time.time()
        
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            for train_iter in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):

                current_progress = global_step / num_train_steps
                for _, loss_model, _ in train_objectives:
                    if hasattr(loss_model, 'use_drw'):
                        if use_drw:
                            loss_model.use_drw = (current_progress >= drw_ratio)
                        else:
                            loss_model.use_drw = False

                # check if model parameters keep same
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    loss_model.zero_grad()
                    loss_model.train()

                    loss_weight = train_weights[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        if '_build_prompt_sentence' in dir(dataloaders[train_idx].dataset):
                            dataloaders[train_idx].dataset._build_prompt_sentence()
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    if use_amp:
                        with autocast():
                            loss_model_return = loss_model(**data)
                        loss_value = loss_weight * loss_model_return['loss_value']
                        loss_value = loss_value
                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_model_return = loss_model(**data)
                        loss_value = loss_weight * loss_model_return['loss_value'] / self.accumulation_steps
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    train_loss_dict[train_idx].append(loss_value.item())
                    optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps>0 and global_step % evaluation_steps == 0:
                    print('\n######### Train Loss #########')
                    for key in train_loss_dict.keys():
                        print('{} {:.4f} \n'.format(key, np.mean(train_loss_dict[key])))
                    train_loss_dict = defaultdict(list)

                    # Continuous Hardware & Time Tracking
                    current_process = psutil.Process(os.getpid())
                    ram_bytes = current_process.memory_info().rss
                    for child in current_process.children(recursive=True):
                        try:
                            ram_bytes += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    ram_gb = ram_bytes / (1024**3)
                    
                    if torch.cuda.is_available():
                        vram_current_gb = torch.cuda.memory_allocated(0) / (1024**3)
                        vram_peak_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
                        vram_reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
                        gpu_name = torch.cuda.get_device_name(0)
                        
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(1)
                            power_w = f"{pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0:.1f}W"
                        except:
                            power_w = "N/A"
                    else:
                        vram_current_gb = 0.0
                        vram_peak_gb = 0.0
                        vram_reserved_gb = 0.0
                        gpu_name = "CPU"
                        power_w = "N/A"

                    elapsed_time = time.time() - train_start_time
                    print(f"\n--- Hardware & Time Tracking ---")
                    print(f"Time Elapsed: {elapsed_time/60:.2f} mins | Your CPU RAM: {ram_gb:.2f}GB | Device: {gpu_name}")
                    print(f"VRAM Peak: {vram_peak_gb:.2f}GB | VRAM Reserved: {vram_reserved_gb:.2f}GB | VRAM Static: {vram_current_gb:.2f}GB | Power: {power_w}\n")

                if evaluation_steps > 0 and global_step % evaluation_steps == 0 and self.evaluator is not None:
                    scores = self.evaluator.evaluate()
                    print(f'\n######### Eval {global_step} #########')
                    
                    # 1. Log Hardware & Time
                    self.score_logs['global_step'].append(global_step)
                    self.score_logs['time_mins'].append(elapsed_time / 60)
                    self.score_logs['ram_gb'].append(ram_gb)
                    self.score_logs['vram_peak_gb'].append(vram_peak_gb)
                    self.score_logs['vram_reserved_gb'].append(vram_reserved_gb)
                    self.score_logs['vram_static_gb'].append(vram_current_gb)
                    
                    # 2. Log ALL Metrics from Evaluator
                    for key in scores.keys():
                        print('{}: {:.4f}'.format(key, scores[key]))
                        self.score_logs[key].append(scores[key])
                    
                    # 3. Save to a persistent CSV file
                    import pandas as pd
                    os.makedirs(output_path, exist_ok=True)
                    pd.DataFrame(self.score_logs).to_csv(os.path.join(output_path, 'training_metrics_and_hardware.csv'), index=False)

                    if global_step == 8000:
                        save_dir = os.path.join(output_path, f'{global_step}/')
                        self._save_ckpt(model, save_dir)

                if self.evaluator is None and global_step == 8000:
                    state_dict = model.state_dict()
                    save_dir = os.path.join(output_path, f'{global_step}/')
                    self._save_ckpt(model, save_dir)
                    print('model saved to', os.path.join(output_path, WEIGHTS_NAME))

        if save_best_model:
            import pandas as pd
            from distutils.dir_util import copy_tree
            res = pd.DataFrame(self.score_logs)
            if 'global_step' in res.columns:
                res = res.set_index('global_step')
                
                # take the average column best (excluding hardware/time stats from the average!)
                eval_cols = [c for c in res.columns if c not in ['time_mins', 'ram_gb', 'vram_peak_gb', 'vram_reserved_gb', 'vram_static_gb']]
                if len(eval_cols) > 0:
                    best_iter = res[eval_cols].mean(1).idxmax()
                    best_save_path = os.path.join(output_path, './best')
                    if not os.path.exists(best_save_path): os.makedirs(best_save_path)
                    best_origin_path = os.path.join(output_path, f'./{best_iter}')
                    if os.path.exists(best_origin_path):
                        print(f'save best checkpoint at iter {best_iter} to', best_save_path)
                        copy_tree(best_origin_path, best_save_path)
            else:
                print("Skipping 'save_best_model': No evaluation steps were triggered during training.")

        if eval_dataloader is not None and load_best_model_at_last and save_best_model and evaluator is not None:
            if 'best_save_path' in locals() and os.path.exists(best_save_path):
                state_dict = torch.load(os.path.join(best_save_path, WEIGHTS_NAME))
                model.load_state_dict(state_dict)
                print(f'load best checkpoint at last from {best_save_path}')

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _save_ckpt(self, model, save_dir):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, WEIGHTS_NAME))