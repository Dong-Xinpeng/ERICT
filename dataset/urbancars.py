"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import glob
import torch
import random
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class UrbanCars(Dataset):

    obj_name_list = [
        "urban",
        "country",
    ]

    bg_name_list = [
        "urban",
        "country",
    ]

    co_occur_obj_name_list = [
        "urban",
        "country",
    ]

    def __init__(
        self,
        cfg,
        split: str,
        transform=None,
        group_label="bg",
        return_group_index=False,
        return_domain_label=False,
        return_dist_shift=False,
    ):
        if split == "train":
            bg_ratio = 0.95
            co_occur_obj_ratio = 0.95
        elif split in ["val", "test"]:
            bg_ratio = 0.5
            co_occur_obj_ratio = 0.5
        else:
            raise NotImplementedError
        self.cfg = cfg
        self.split = split
        self.bg_ratio = bg_ratio
        self.co_occur_obj_ratio = co_occur_obj_ratio


        super().__init__()
        assert group_label in ["bg", "co_occur_obj", "both"]
        self.transform = transform
        self.return_group_index = return_group_index
        self.return_domain_label = return_domain_label
        self.return_dist_shift = return_dist_shift

        self.root = cfg.root_dir

        ratio_combination_folder_name = (
            f"bg-{bg_ratio}_co_occur_obj-{co_occur_obj_ratio}"
        )
        img_root = os.path.join(
            self.root, ratio_combination_folder_name, split
        )

        print('img_root:',img_root)

        self.img_fpath_list = []
        self.obj_bg_co_occur_obj_label_list = []

        for obj_id, obj_name in enumerate(self.obj_name_list):
            for bg_id, bg_name in enumerate(self.bg_name_list):
                for co_occur_obj_id, co_occur_obj_name in enumerate(self.co_occur_obj_name_list):
                    
                    dir_name = (
                        f"obj-{obj_name}_bg-{bg_name}_co_occur_obj-{co_occur_obj_name}"
                    )
                    dir_path = os.path.join(img_root, dir_name)
                    assert os.path.exists(dir_path)

                    img_fpath_list = glob.glob(os.path.join(dir_path, "*.jpg"))
                    self.img_fpath_list += img_fpath_list

                    self.obj_bg_co_occur_obj_label_list += [
                        (obj_id, bg_id, co_occur_obj_id)
                    ] * len(img_fpath_list)

        self.obj_bg_co_occur_obj_label_list = torch.tensor(
            self.obj_bg_co_occur_obj_label_list, dtype=torch.long
        )

        self.obj_label = self.obj_bg_co_occur_obj_label_list[:, 0]
        self.bg_label = self.obj_bg_co_occur_obj_label_list[:, 1]
        self.co_occur_obj_label = self.obj_bg_co_occur_obj_label_list[:, 2]

        if group_label == "bg":
            num_shortcut_category = 2
            shortcut_label = self.bg_label
     
        elif group_label == "co_occur_obj":
            num_shortcut_category = 2
            shortcut_label = self.co_occur_obj_label

        elif group_label == "both":
            num_shortcut_category = 4
            shortcut_label = self.bg_label * 2 + self.co_occur_obj_label

        else:
            raise NotImplementedError

        self.spurious_label = shortcut_label
        self.set_num_group_and_group_array(num_shortcut_category, shortcut_label)


        self.n_classes = len(self.obj_name_list)
        self.y_array_onehot = torch.zeros(len(self.obj_label), self.n_classes)
        self.y_array_onehot = self.y_array_onehot.scatter_(1, self.obj_label.unsqueeze(1), 1)

        self.n_groups = self.num_group


        # print(self.obj_label)
        # print(len(self.obj_label))
        # print(len(self.img_fpath_list))
        # print(len(self.group_array))

        # print(type(self.obj_label))
        # print(type(self.img_fpath_list))
        # print(type(self.group_array))


        # print(self.obj_label[0])
        # print(self.img_fpath_list[0])
        # print(self.group_array[0])
        # assert False



    def set_num_group_and_group_array(self, num_shortcut_category, shortcut_label):
        self.num_group = len(self.obj_name_list) * num_shortcut_category
        self.group_array = self.obj_label * num_shortcut_category + shortcut_label


    def set_domain_label(self, shortcut_label):
        self.domain_label = shortcut_label

    def __len__(self):
        return len(self.img_fpath_list)

    def __getitem__(self, idx):
        img_fpath = self.img_fpath_list[idx]
        y = self.obj_label[idx]
        group = self.group_array[idx]
        y_onehot = self.y_array_onehot[idx]


        img = Image.open(img_fpath)
        img = img.convert("RGB")


        if (self.cfg.view) or (self.cfg.mask_mode == 'img_token'):
            x_view = self.temp_debug(img)  # todo
            x = self.transform(img) 
            return (x, y, y_onehot, group, idx, x_view) 
        
        else:
            x = self.transform(img) # todo
            return (x, y, y_onehot, group, idx) 

 

    def get_labels(self):
        return self.obj_bg_co_occur_obj_label_list

    def get_sampling_weights(self):
        group_counts = (
            (torch.arange(self.num_group).unsqueeze(1) == self.group_array)
            .sum(1)
            .float()
        )
        group_weights = len(self) / group_counts
        weights = group_weights[self.group_array]
        return weights

    def temp_debug(self,x):
    
        transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
        ])

        return transform(x)

    def _get_subsample_group_indices(self, subsample_which_shortcut):
        bg_ratio = self.bg_ratio
        co_occur_obj_ratio = self.co_occur_obj_ratio

        num_img_per_obj_class = len(self) // len(self.obj_name_list)
        if subsample_which_shortcut == "bg":
            min_size = int(min(1 - bg_ratio, bg_ratio) * num_img_per_obj_class)
        elif subsample_which_shortcut == "co_occur_obj":
            min_size = int(min(1 - co_occur_obj_ratio, co_occur_obj_ratio) * num_img_per_obj_class)
        elif subsample_which_shortcut == "both":
            min_bg_ratio = min(1 - bg_ratio, bg_ratio)
            min_co_occur_obj_ratio = min(1 - co_occur_obj_ratio, co_occur_obj_ratio)
            min_size = int(min_bg_ratio * min_co_occur_obj_ratio * num_img_per_obj_class)
        else:
            raise NotImplementedError

        assert min_size > 1

        indices = []

        if subsample_which_shortcut == "bg":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = self.obj_bg_co_occur_obj_label_list[:, 0] == idx_obj
                for idx_bg in range(len(self.bg_name_list)):
                    bg_mask = self.obj_bg_co_occur_obj_label_list[:, 1] == idx_bg
                    mask = obj_mask & bg_mask
                    subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                    random.shuffle(subgroup_indices)
                    sampled_subgroup_indices = subgroup_indices[:min_size]
                    indices += sampled_subgroup_indices
        elif subsample_which_shortcut == "co_occur_obj":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = self.obj_bg_co_occur_obj_label_list[:, 0] == idx_obj
                for idx_co_occur_obj in range(len(self.co_occur_obj_name_list)):
                    co_occur_obj_mask = self.obj_bg_co_occur_obj_label_list[:, 2] == idx_co_occur_obj
                    mask = obj_mask & co_occur_obj_mask
                    subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                    random.shuffle(subgroup_indices)
                    sampled_subgroup_indices = subgroup_indices[:min_size]
                    indices += sampled_subgroup_indices
        elif subsample_which_shortcut == "both":
            for idx_obj in range(len(self.obj_name_list)):
                obj_mask = self.obj_bg_co_occur_obj_label_list[:, 0] == idx_obj
                for idx_bg in range(len(self.bg_name_list)):
                    bg_mask = self.obj_bg_co_occur_obj_label_list[:, 1] == idx_bg
                    for idx_co_occur_obj in range(len(self.co_occur_obj_name_list)):
                        co_occur_obj_mask = self.obj_bg_co_occur_obj_label_list[:, 2] == idx_co_occur_obj
                        mask = obj_mask & bg_mask & co_occur_obj_mask
                        subgroup_indices = torch.nonzero(mask).squeeze().tolist()
                        random.shuffle(subgroup_indices)
                        sampled_subgroup_indices = subgroup_indices[:min_size]
                        indices += sampled_subgroup_indices
        else:
            raise NotImplementedError

        return indices


def load_urbancars(cfg, train_shuffle=True, transform=None):

    batch_size = cfg.batch_size
   
    # train_set = MetaDatasetCatDog(cfg,split='train', train_transform=transform)
    # train_loader = DataLoader(train_set, batch_size=batch_size,
    #                         shuffle=train_shuffle,
    #                         num_workers=cfg.num_workers)

    # val_set = MetaDatasetCatDog(cfg,split='val', train_transform=transform)
    # val_loader = DataLoader(val_set, batch_size=batch_size,
    #                         shuffle=False, num_workers=cfg.num_workers)

    test_set = UrbanCars(cfg,split='test', transform=transform)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, num_workers=cfg.num_workers)
    
    cfg.num_classes = 2
    cfg.num_groups = test_set.n_groups 
    # assert False, cfg.num_groups

    return {'train':None, 'val':None, 'test':test_loader}


def load_dataloaders(cfg, train_shuffle=True, 
                     val_correlation=None,
                     transform=None):
    return load_urbancars(cfg, train_shuffle, transform)