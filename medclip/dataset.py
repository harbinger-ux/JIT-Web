import re
import random
from collections import defaultdict
import pdb
from typing import Union, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torchvision import transforms

from transformers import AutoTokenizer
from transformers import CLIPFeatureExtractor, CLIPProcessor
from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import is_torch_tensor

import nltk
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder

from .prompts import process_class_prompts, process_class_prompts_for_tuning
from .prompts import generate_chexpert_class_prompts
from . import constants

class MedCLIPFeatureExtractor(CLIPFeatureExtractor):
    def __init__(self, 
        do_resize=True, 
        size=224, 
        resample=Image.BICUBIC, 
        do_center_crop=True, 
        crop_size=224, 
        do_normalize=True, 
        image_mean=constants.IMG_MEAN, 
        image_std=constants.IMG_STD, 
        do_convert_rgb=False,
        do_pad_square=True,
        **kwargs):
        super().__init__(do_resize, size, resample, do_center_crop, crop_size, do_normalize, image_mean, image_std, do_convert_rgb, **kwargs)
        self.do_pad_square = do_pad_square
    
    def __call__(self, 
        images: Union[Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]], 
        return_tensors: Optional[Union[str, TensorType]] = None, 
        **kwargs) -> BatchFeature:

        valid_images = False
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor`"
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # Base transformations (RGB + padding)
        if self.do_convert_rgb:
            images = [image.convert("RGB") if hasattr(image, "convert") else image for image in images]

        if self.do_pad_square:
            images = [self.pad_img(image,min_size=self.size) for image in images]
        
        # Allowing PIL images to natively pass through resize/crop for older transformers compatibility
        if self.do_resize and self.size is not None and self.resample is not None:
            images = [
                self.resize(image=image, size=self.size, resample=self.resample)
                for image in images
            ]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
            
        # Normalization with safe dimension handling
        if self.do_normalize:
            images_normalized = []
            for image in images:
                # Convert to numpy array safely just to check channels
                tmp = np.array(image) if isinstance(image, Image.Image) else image
                c = tmp.shape[-1] if tmp.ndim == 3 else 1
                
                # Truncate mean/std to match single-channel images
                mean = self.image_mean[:c] if isinstance(self.image_mean, (list, tuple)) else self.image_mean
                std = self.image_std[:c] if isinstance(self.image_std, (list, tuple)) else self.image_std
                
                images_normalized.append(self.normalize(image=image, mean=mean, std=std))
            images = images_normalized
        
        # Final conversion to array and dimension expansion
        images = [np.array(image) if isinstance(image, Image.Image) else image for image in images]

        images_ = []
        for image in images:
            if len(image.shape) == 2:
                image = image[None]
            images_.append(image)
        images = images_

        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def pad_img(self, img, min_size=224, fill_color=0):
        x, y = img.size
        if isinstance(min_size, dict):
            min_size = min_size.get("shortest_edge", min_size.get("height", 224))
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

class MedCLIPProcessor(CLIPProcessor):
    feature_extractor_class = "CLIPFeatureExtractor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")
    def __init__(self):
        feature_extractor = MedCLIPFeatureExtractor()
        tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        tokenizer.model_max_length = 77
        super().__init__(feature_extractor, tokenizer)

# STREAMING_CHUNK: True Mixup Implementation (Linear Blending)
class ImageTextContrastiveDataset(Dataset):
    def __init__(self, img_csv_path, txt_csv_path, class_names, imgtransform=None, G=0.001, use_mixup=True) -> None:
        super().__init__()
        self._labels_ = class_names
        self.use_mixup = use_mixup
        
        print('Loading image data from', img_csv_path)
        self.img_df = pd.read_csv(img_csv_path).fillna(0)
        
        print('Loading text data from', txt_csv_path)
        self.txt_df = pd.read_csv(txt_csv_path).fillna(0)

        # Robust text column finder
        self.txt_col = 'sentence'
        for col in self.txt_df.columns:
            if col.strip().lower() in ['sentence', 'report', 'text', 'caption']:
                self.txt_col = col
                break
                
        # Robust image path column finder
        self.img_col = 'image_path'
        for col in self.img_df.columns:
            if col.strip().lower() in ['image_path', 'imgpath', 'path', 'image path', 'img_path']:
                self.img_col = col
                break
        
        # ALWAYS decouple using G parameter
        self.m = len(self.img_df)
        self.n = len(self.txt_df)
        self.G = G
        self.epoch_length = max(1, int(self.G * (self.m * self.n)))

        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])],
            )
        else:
            self.transform = imgtransform

    def __getitem__(self, index):
        # 1. Base Image Sampling
        img_index = index % self.m
        img_row = self.img_df.iloc[img_index]
        
        img = Image.open(img_row[self.img_col])
        img = self._pad_img(img) 
        img = self.transform(img).unsqueeze(1)
        img_label = img_row[self._labels_].values.astype(float)
        
        # 2. Base Text Random Sampling
        txt_index = random.randint(0, self.n - 1)
        txt_row = self.txt_df.iloc[txt_index]
        report = txt_row[self.txt_col] 
        text_label = txt_row[self._labels_].values.astype(float)

        # 3. MixUp Data Augmentation (Linear Blending)
        if self.use_mixup:
            # Sample mixing coefficient from Beta distribution
            lam = np.random.beta(0.5, 0.5)
            
            # Sample a second image
            idx2 = random.randint(0, self.m - 1)
            row2 = self.img_df.iloc[idx2]
            img2 = Image.open(row2[self.img_col])
            img2 = self._pad_img(img2)
            img2 = self.transform(img2).unsqueeze(1)
            img_label2 = row2[self._labels_].values.astype(float)
            
            # Mathematically blend the image tensors and their labels
            img = lam * img + (1.0 - lam) * img2
            img_label = lam * img_label + (1.0 - lam) * img_label2
            
            # Sample a second text
            txt_idx2 = random.randint(0, self.n - 1)
            txt_row2 = self.txt_df.iloc[txt_idx2]
            report2 = txt_row2[self.txt_col]
            text_label2 = txt_row2[self._labels_].values.astype(float)
            
            # Concatenate text strings so the text encoder sees both, and blend their labels
            report = str(report) + " " + str(report2)
            text_label = lam * text_label + (1.0 - lam) * text_label2

        return img, report, img_label, text_label

    def __len__(self):
        return self.epoch_length

    def _pad_img(self, img, min_size=224, fill_color=0):
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im


class ImageTextContrastiveCollator:
    def __init__(self, use_eda=True):
        if use_eda:
            import nltk
            nltk.download('stopwords')
            nltk.download('omw-1.4')
            nltk.download('wordnet')
            from textaugment import EDA
            self.eda = EDA()
        else:
            self.eda = None

        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
        
    def __call__(self, batch):
        inputs = defaultdict(list)
        report_list = []
        report_aug_list = []
        for data in batch:
            inputs['pixel_values'].append(data[0])
            if self.eda is not None:
                eda_aug = random.choice([self.eda.synonym_replacement, self.eda.random_swap, self.eda.random_deletion])
                text_aug = eda_aug(data[1])
                if isinstance(text_aug, list): text_aug = ' '.join(text_aug)
                report_aug_list.append(text_aug)
            report_list.append(data[1])
            inputs['img_labels'].append(data[2])
            inputs['text_labels'].append(data[3])
            
        text_inputs = self.tokenizer(report_list, truncation=True, padding=True, return_tensors='pt')
        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        
        inputs['img_labels'] = torch.tensor(np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(np.stack(inputs['text_labels']).astype(float))
        inputs['input_ids'] = text_inputs['input_ids']
        inputs['attention_mask'] = text_inputs['attention_mask']
        
        if len(report_aug_list) > 0:
            aug_text_inputs = self.tokenizer(report_aug_list, truncation=True, padding=True, return_tensors='pt')
            inputs['aug_input_ids'] =  aug_text_inputs['input_ids']
            inputs['aug_attention_mask'] = aug_text_inputs['attention_mask']

        return inputs

class ZeroShotImageDataset(Dataset):
    def __init__(self, datalist=['chexpert-5x200'], class_names=None, imgtransform=None) -> None:
        super().__init__()
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform
        self.class_names = class_names
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
        
        self.img_col = 'image_path'
        for col in self.df.columns:
            if col.strip().lower() in ['image_path', 'imgpath', 'path', 'image path', 'img_path']:
                self.img_col = col
                break

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row[self.img_col])
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)

class ZeroShotImageCollator:
    def __init__(self, mode, cls_prompts=None, n_prompt=5):
        self.tokenizer = AutoTokenizer.from_pretrained(constants.BERT_TYPE)
        self.tokenizer.model_max_length = 77
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode
        if cls_prompts is None:
            raise NotImplementedError
        else:
            self.cls_prompts = cls_prompts
        self.prompt_texts_inputs = process_class_prompts(self.cls_prompts)

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])

        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values
        if self.mode in ['multiclass','binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        return {
            'pixel_values': inputs['pixel_values'],
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
            }

class SuperviseImageDataset(Dataset):
    def __init__(self, datalist=['chexpert-5x200'], class_names=None, imgtransform=None) -> None:
        super().__init__()
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE,constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838],std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform
        self.class_names = class_names
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
        
        self.img_col = 'image_path'
        for col in self.df.columns:
            if col.strip().lower() in ['image_path', 'imgpath', 'path', 'image path', 'img_path']:
                self.img_col = col
                break

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row[self.img_col])
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)

class SuperviseImageCollator:
    def __init__(self, mode):
        assert mode in ['multiclass','multilabel','binary']
        self.mode = mode

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])
        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values

        if self.mode in ['multiclass','binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1,3,1,1))
        return {
            'pixel_values': inputs['pixel_values'],
            'labels': inputs['labels'],
            }

class PromptTuningImageDataset(Dataset):
    def __init__(self, datalist=['chexpert-5x200'], class_names=None, imgtransform=None) -> None:
        super().__init__()
        if imgtransform is None:
            self.transform = transforms.Compose([
                transforms.Resize((constants.IMG_SIZE, constants.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5862785803043838], std=[0.27950088968644304])]
            )
        else:
            self.transform = imgtransform
        self.class_names = class_names
        df_list = []
        for data in datalist:
            filename = f'./local_data/{data}-meta.csv'
            df = pd.read_csv(filename, index_col=0)
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)
        
        self.img_col = 'image_path'
        for col in self.df.columns:
            if col.strip().lower() in ['image_path', 'imgpath', 'path', 'image path', 'img_path']:
                self.img_col = col
                break

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(row[self.img_col])
        img = self._pad_img(img)
        img = self.transform(img).unsqueeze(1)
        label = pd.DataFrame(row[self.class_names]).transpose()
        return img, label

    def _pad_img(self, img, min_size=224, fill_color=0):
        x, y = img.size
        size = max(min_size, x, y)
        new_im = Image.new('L', (size, size), fill_color)
        new_im.paste(img, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def __len__(self):
        return len(self.df)

class PromptTuningImageCollator:
    def __init__(self, mode, cls_prompts=None, n_prompt=5, n_context=16, class_specific_context=False):
        assert mode in ['multiclass', 'multilabel', 'binary']
        self.mode = mode
        if cls_prompts is None:
            raise NotImplementedError
        else:
            self.cls_prompts = cls_prompts
        self.prompt_texts_inputs = process_class_prompts_for_tuning(self.cls_prompts, n_context=n_context, class_specific_context=class_specific_context)

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data[0])
            inputs['labels'].append(data[1])

        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values
        if self.mode in ['multiclass', 'binary']:
            inputs['labels'] = torch.tensor(inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.cat(inputs['pixel_values'], 0)
        if inputs['pixel_values'].shape[1] == 1: inputs['pixel_values'] = inputs['pixel_values'].repeat((1, 3, 1, 1))
        return {
            'pixel_values': inputs['pixel_values'],
            'prompt_inputs': self.prompt_texts_inputs,
            'labels': inputs['labels'],
        }