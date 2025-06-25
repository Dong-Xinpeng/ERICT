"""
Waterbirds dataset
- Reference code: https://github.com/kohpangwei/group_DRO/blob/master/data/cub_dataset.py
- See Group DRO, https://arxiv.org/abs/1911.08731 for more
"""
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from dataset import train_val_split, plot_data_batch



class Waterbirds(Dataset):
    """
    Waterbirds dataset from waterbird_complete95_forest2water2 in Group DRO paper
    """
        
    def __init__(self,cfg,split,augment_data=False,train_transform = None, split2groupid = -1):
        self.cfg = cfg
        self.root_dir = cfg.root_dir
        self.target_name = cfg.target_name
        self.confounder_names = cfg.confounder_names

        self.augment_data = augment_data
        self.split = split

        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.data_dir = self.root_dir

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, 'metadata.csv'))
        # Filter for data split ('train', 'val', 'test')
        self.metadata_df = self.metadata_df[
            self.metadata_df['split'] == self.split_dict[self.split]]
        

        # Get the y values
        self.y_array = self.metadata_df['y'].values
        self.n_classes = 2

        # print(type(self.y_array))  # 
        self.y_array_onehot = torch.zeros(len(self.y_array), self.n_classes) # [n,2]
        self.y_array_onehot = self.y_array_onehot.scatter_(1, torch.tensor(self.y_array, dtype=torch.int64).unsqueeze(1), 1).numpy()

        # assert False

        

        # We only support one confounder for CUB for now
        self.confounder_array = self.metadata_df['place'].values
        self.n_confounders = 1

        # Map to groups
        self.n_groups = pow(2, 2)
        self.group_array = (self.y_array * (self.n_groups / 2) +
                            self.confounder_array).astype('int')
        
        

        # Extract filenames and splits
        self.filename_array = self.metadata_df['img_filename'].values
        # self.split_array = self.metadata_df['split'].values

        # self.targets = torch.tensor(self.y_array)
        
        self.group_labels = ['LANDBIRD on land', 'LANDBIRD on water',
                             'WATERBIRD on land', 'WATERBIRD on water']
        self.class_names = ['LANDBIRD', 'WATERBIRD']

        if cfg.dataset == 'waterbirds_r':
            self.group_labels = ['LAND with landbird', 'LAND with waterbird',
                                 'WATER with landbird', 'WATER with waterbird']

        # Set transform
        self.train_transform = train_transform
        self.eval_transform = train_transform

        if self.split == 'train' and split2groupid >= 0 :
          self.split_group(split2groupid)
        

        self.targets_all = {'target': np.array(self.y_array),
                            'group_idx': np.array(self.group_array),
                            'spurious': np.array(self.confounder_array),
                            'sub_target': np.array(list(zip(self.y_array, self.confounder_array)))}


        

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]  
        group = self.group_array[idx]
        # spurious = self.targets_all['spurious'][idx]
        y_onehot = self.y_array_onehot[idx]

        img_filename = os.path.join(
            self.data_dir,
            self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')

    
        x = self.train_transform(img) 
        return (x, y, y_onehot, group, idx) 


    


def load_waterbirds(cfg, train_shuffle=True, transform=None):

    batch_size = cfg.batch_size

    train_set = Waterbirds(cfg,split='train', train_transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=train_shuffle,
                            num_workers=cfg.num_workers)

    val_set = Waterbirds(cfg,split='val', train_transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=cfg.num_workers)

    test_set = Waterbirds(cfg,split='test', train_transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    cfg.num_classes = 2
    cfg.num_groups = 4
    # train is [DataLoader] or DataLoader
    return {'train':train_loader, 'val':val_loader, 'test':test_loader}



def load_dataloaders(cfg, train_shuffle=True, 
                     val_correlation=None,
                     transform=None):
    return load_waterbirds(cfg, train_shuffle, transform)





