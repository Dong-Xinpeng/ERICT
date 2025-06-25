"""
CelebA Dataset
- Reference code: https://github.com/kohpangwei/group_DRO/blob/master/data/celebA_dataset.py
- See Group DRO, https://arxiv.org/abs/1911.08731 for more
"""
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CelebA(Dataset):
    _normalization_stats = {'mean': (0.485, 0.456, 0.406), 
                            'std': (0.229, 0.224, 0.225)}

    def __init__(self, cfg, split='train', augment_data=False,train_transform=None,split2groupid = -1):
        self.cfg = cfg
        self.root_dir = cfg.root_dir
        self.target_name = cfg.target_name
        self.confounder_names = cfg.confounder_names 
        # Only support 1 confounder for now as in official benchmark
        confounder_names = self.confounder_names[0]  

        self.augment_data = augment_data
        self.split = split

        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }
        
        self.data_dir = os.path.join(self.root_dir,'celebA')

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        # Read in metadata
        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, 'list_attr_celeba.csv'))#, delim_whitespace=True)
        self.split_df = pd.read_csv(os.path.join(self.data_dir, 'list_eval_partition.csv'))#, delim_whitespace=True)
        # Filter for data split ('train', 'val', 'test')
        self.metadata_df['partition'] = self.split_df['partition']
        self.metadata_df = self.metadata_df[
            self.split_df['partition'] == self.split_dict[self.split]]
        
        if self.split == 'train':
            self.metadata_df = self.metadata_df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True) 
        
      

        # Get the y values
        self.y_array = self.metadata_df[self.target_name].values
        self.confounder_array = self.metadata_df[confounder_names].values
        self.y_array[self.y_array == -1] = 0
        self.confounder_array[self.confounder_array == -1] = 0
        self.n_classes = len(np.unique(self.y_array))
        self.n_confounders = len(confounder_names)

        self.y_array_onehot = torch.zeros(len(self.y_array), self.n_classes)
        self.y_array_onehot = self.y_array_onehot.scatter_(1, torch.tensor(self.y_array, dtype=torch.int64).unsqueeze(1), 1).numpy()

        
        # Get sub_targets / group_idx
        self.metadata_df['sub_target'] = (
            self.metadata_df[self.target_name].astype(str) + '_' +
            self.metadata_df[confounder_names].astype(str))
        
        # Get subclass map
        attributes = [self.target_name, confounder_names]
        self.df_groups = (self.metadata_df[
            attributes].groupby(attributes).size().reset_index())
        self.df_groups['group_id'] = (
            self.df_groups[self.target_name].astype(str) + '_' +
            self.df_groups[confounder_names].astype(str))
        self.subclass_map = self.df_groups[
            'group_id'].reset_index().set_index('group_id').to_dict()['index']
        self.group_array = self.metadata_df['sub_target'].map(self.subclass_map).values
        groups, group_counts = np.unique(self.group_array, return_counts=True)
        self.n_groups = len(groups)

        # Extract filenames and splits
        self.filename_array = self.metadata_df['image_id'].values
        # self.split_array = self.metadata_df['partition'].values


        
        # Image transforms
        if train_transform is not None:
            self.train_transform = train_transform
        else:
            assert False, "train_transform must be provided"

        
        # self.show_data_distribution()

        self.targets_all = {'target': np.array(self.y_array),
                            'group_idx': np.array(self.group_array),
                            'spurious': np.array(self.confounder_array),
                            'sub_target': np.array(list(zip(self.y_array, self.confounder_array)))}
        

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx] 
        group = self.group_array[idx]
        y_onehot = self.y_array_onehot[idx]

        img_filename = os.path.join(
            self.data_dir,
            'img_align_celeba',
            'img_align_celeba',
            self.filename_array[idx])
        img = Image.open(img_filename)
        # Figure out split and transform accordingly
        
        
        x = self.train_transform(img) # todo
        return (x, y, y_onehot, group, idx) 
    


    


def load_celeba(cfg, train_shuffle=True, transform=None):
    batch_size = cfg.batch_size

    # train_set = CelebA(cfg, split='train', train_transform=transform)
    # train_loader = DataLoader(train_set, batch_size=batch_size,
    #                           shuffle=train_shuffle, num_workers=cfg.num_workers)

    # val_set = CelebA(cfg, split='val', train_transform=transform)
    # val_loader = DataLoader(val_set, batch_size=batch_size,
    #                         shuffle=False, num_workers=cfg.num_workers)
    
    test_set = CelebA(cfg, split='test', train_transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    cfg.num_classes = 2
    cfg.num_groups = 4
    # return (train_loader, val_loader, test_loader)
    return {'train':None, 'val':None, 'test':test_loader}




   
def load_dataloaders(cfg, train_shuffle=True, 
                     val_correlation=None,
                     transform=None):
    return load_celeba(cfg, train_shuffle, transform)

