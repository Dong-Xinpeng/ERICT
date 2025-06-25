

import torch
import torch.nn as nn
import os
from PIL import Image
from torch.nn import functional as F
from clip_twice import clip
import numpy as np
import json

from prompt.imagenet_templates import IMAGENET_TEMPLATES
from prompt.template import * 
from prompt.imagenet_templates import *


def load_clip(cfg):
    backbone = cfg.backbone  # example: 'RN50'

    print(backbone)
    base_model, transformer = clip.load(backbone)

    return base_model, transformer


def build_model(cfg):
    model,transform = load_clip(cfg)
    return model,transform



class zsclip_twice(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.device = cfg.device
        self.cfg = cfg
        self.model, self.transformer = build_model(cfg)
    
        for param in self.model.parameters():
            param.requires_grad = False

        self.img_encoder = self.model.visual
        self.txt_encoder = self.model.encode_text


        
        if cfg.dataset == 'waterbirds':
            prompts = waterbirds_prompts
            class_name_list = waterbirds_class_names
        elif cfg.dataset == 'celebA':
            prompts = celebA_prompts
            class_name_list = celebA_class_names
        elif cfg.dataset == 'urbancars':
            prompts = urbancars_prompts
            class_name_list = urbancars_class_names
        else:
            assert False
        

        self.help_prompt = cfg.help_prompt
        

        self.ratio_bar = cfg.ratio_bar
       
        # class prompt

        cfg.prompts = prompts
        text_tokens = clip.tokenize(prompts).to(self.device)
        text_features = self.txt_encoder(text_tokens)  # raw clip
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.logit_scale = self.model.logit_scale


        if self.help_prompt is not None:
            bias_text_tokens = clip.tokenize(self.help_prompt).to(self.device)
            help_text_features = self.txt_encoder(bias_text_tokens)  # raw clip
            self.help_text_features = help_text_features / help_text_features.norm(dim=-1, keepdim=True)
        else:
            self.help_text_features = None

        
        self.mask_mode = cfg.mask_mode
        self.score_mode = cfg.score_mode

        # class name list 
        class_name_text_tokens = clip.tokenize(class_name_list).to(self.device)
        class_name_text_features = self.txt_encoder(class_name_text_tokens)  # raw clip


    
        if self.help_prompt == '_class':
            class_name_text_features = class_name_text_features.mean(dim=0)
            class_name_text_features = class_name_text_features / class_name_text_features.norm(dim=-1, keepdim=True)
            class_name_text_features = class_name_text_features.unsqueeze(0)
            self.help_text_features = class_name_text_features
            print(f'renew help_text_features to class_name_text_features')
        
        
        elif '_class_all' in self.help_prompt:
            # class_name_text_features = class_name_text_features.mean(dim=0)
            class_name_text_features = class_name_text_features / class_name_text_features.norm(dim=-1, keepdim=True)
            self.help_text_features = class_name_text_features  # [class_num, dim]
            print(f'renew help_text_features to _class_all')
    
        
    


    def forward(self,image):

        image_features, img_spatial_feat = self.img_encoder(image, self.help_text_features, cfg = self.cfg)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * image_features @ self.text_features.t()
        return logits
    





