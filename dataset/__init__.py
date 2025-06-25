"""
Datasets and data utils

Functions:
- initialize_data()
- train_val_split()
- get_resampled_set()
- imshow()
- plot_data_batch()
"""
import copy
import os
import json
import numpy as np
import importlib
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid, save_image


def initialize_data(args, mode = None):

    print("dataset:",args.dataset)

    
    dataset_module = importlib.import_module(f'dataset.{args.dataset}')
    load_dataloaders = getattr(dataset_module, 'load_dataloaders')

    

    if args.dataset == 'waterbirds':
        args.root_dir = os.path.join(args.root_dir, 'waterbirds')
        args.val_split = 0.2
        args.target_name = 'waterbird_complete95'
        args.confounder_names = ['forest2water2']
        args.augment_data = False
        args.train_classes = ['landbird', 'waterbird']
        ## Image
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        # args.text_descriptions = ['a landbird', 'a waterbird']
        args.wilds_dataset = False
        args.num_workers = 4
        
    elif 'celebA' == args.dataset:
        args.target_name = 'Blond_Hair'
        args.confounder_names = ['Male']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.train_classes = ['nonblond', 'blond']
        args.val_split = 0.2

        args.wilds_dataset = False
        args.num_workers = 4
    
    elif 'urbancars' == args.dataset:
        args.root_dir = os.path.join(args.root_dir, 'urbancars')
        args.num_workers = 4

    else:
        raise NotImplementedError
    

    return load_dataloaders




def train_val_split(dataset, val_split, seed):
    """
    Compute indices for train and val splits
    
    Args:
    - dataset (torch.utils.data.Dataset): Pytorch dataset
    - val_split (float): Fraction of dataset allocated to validation split
    - seed (int): Reproducibility seed
    Returns:
    - train_indices, val_indices (np.array, np.array): Dataset indices
    """
    train_ix = int(np.round(val_split * len(dataset)))
    all_indices = np.arange(len(dataset))
    np.random.seed(seed)
    np.random.shuffle(all_indices)
    train_indices = all_indices[train_ix:]
    val_indices = all_indices[:train_ix]
    return train_indices, val_indices


def imshow(img, mean=0.5, std=0.5):
    """
    Visualize data batches
    """
    img = img * std + mean
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_data_batch(dataset, mean=0.0, std=1.0, nrow=8, title=None,
                    args=None, save=False, save_id=None, ftype='png'):
    """
    Visualize data batches
    """
    try:
        img = make_grid(dataset, nrow=nrow)
    except Exception as e:
        raise e
        print(f'Nothing to plot!')
        return
    img = img * std + mean
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    if title is not None:
        plt.title(title)
    if save:
        try:
            fpath = os.path.join(args.image_path,
                         f'{save_id}-{args.experiment_name}.{ftype}')
            plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
        except Exception as e:
            fpath = f'{save_id}-{args.experiment_name}.{ftype}'
            plt.savefig(fname=fpath, dpi=300, bbox_inches="tight")
    if args.display_image:
        plt.show()
    plt.close()


