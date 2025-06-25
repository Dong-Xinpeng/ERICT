import argparse
import torch
import time
import os
from dataset import initialize_data
from train import *

import random
import numpy as np
import warnings
# 忽略所有的警告
warnings.filterwarnings('ignore')
# Pretrained models
from zsclip_twice import zsclip_twice
from zsclip import zsclip



def main(cfg):

    if cfg.model == 'zsclip_twice':
        my_model = zsclip_twice(cfg).to('cuda')
    elif cfg.model == 'zsclip':
        my_model = zsclip(cfg).to('cuda')
    else:
        assert False, "model not implemented"
    
 

    load_dataloaders = initialize_data(cfg)
    dataloaders_base = load_dataloaders(cfg, train_shuffle=True,transform=my_model.transformer)
    
    test_zs(args,my_model,dataloaders_base)
    
 

    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, default='/home/dongxinpeng/datasets')  # data root dir
    parser.add_argument("--output_dir", type=str, default="/home/dongxinpeng/mmlm/ERICT/output")

    parser.add_argument("--seed", type=int, default=78561)  # useless

    parser.add_argument("--dataset", type=str, default='celebA')  

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--backbone', type=str, default="ViTL14") 
    parser.add_argument('--batch_size', type=int, default=32)


    parser.add_argument('--model', type=str, default='') 


    parser.add_argument('--help_prompt',type=str, default=None) 

    parser.add_argument('--ratio_bar', type=float,default=0.5) 

    parser.add_argument('--mask_mode', type=str,default=None)  # weight_all, weight_last, weight_half, weight_n_end
    parser.add_argument('--score_mode', type=str,default='ratio')  #  ratio_abs, 

   
    parser.add_argument('--logist_topk', type=int,default=-1) 
    parser.add_argument('--tau', type=float, default=0.5) 


    parser.add_argument('--weight_n_layer', type=int, default=-1) 



    args = parser.parse_args()


    assert args.ratio_bar <= 1.0

    

    if args.dataset == 'waterbirds':
        args.dataset_classnames = ['landbird','waterbird']
    elif args.dataset == 'celebA':
        # args.dataset_classnames = ['person with non-blonde hair','person with blonde hair']
        args.dataset_classnames = ['non-blonde hair','blonde hair']
    elif args.dataset == 'cmnist':
        args.dataset_classnames = ['number between 0 and 4','number between 5 and 9']

    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print("seed: ",args.seed)


  
    
    main(args)