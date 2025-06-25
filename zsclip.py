

import torch
import torch.nn as nn
import os
from clip import clip


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



class zsclip(nn.Module):
    def __init__(self,cfg):
        super().__init__()

        self.device = cfg.device
        self.cfg = cfg

        self.model,self.transformer = build_model(cfg)

        
        for param in self.model.parameters():
            param.requires_grad = False

        self.img_encoder = self.model.visual
        self.txt_encoder = self.model.encode_text

     
        

        if cfg.dataset == 'waterbirds':
            prompts = waterbirds_prompts
        elif cfg.dataset == 'celebA':
            prompts = celebA_prompts
        elif cfg.dataset == 'urbancars':
            prompts = urbancars_prompts


        self.bar = cfg.bar
        self.ratio_bar = cfg.ratio_bar
       


        cfg.prompts = prompts

        text_tokens = clip.tokenize(prompts).to(self.device)
        text_features = self.txt_encoder(text_tokens)  # raw clip
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        print(prompts)


        self.logit_scale = self.model.logit_scale


        


    def forward(self,image,y=None,g=None):
        image_features = self.img_encoder(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()

        return logits
        

        

