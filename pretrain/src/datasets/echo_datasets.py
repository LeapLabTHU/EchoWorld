import os
import json
import torch
import random
import numpy as np
from PIL import Image, ImageChops
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from autous import Transformation
from einops import rearrange
import cv2
from copy import deepcopy
from sklearn.model_selection import KFold
import time
import logging
from logging import getLogger

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()

def make_echo_dataset(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    pair=False
):
    if pair:
        dataset = PairDatasetYY(split='train', transform=transform, num_repeat=1000, img_size=384)
    else:
        dataset = txtDatasetYY(split='train', transform=transform, num_repeat=1000, img_only=True, img_size=384)

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('ImageNet unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class txtDatasetYY(Dataset):
    def __init__(self, split='train', data_dir='cardiac_dataset/modified_dataset', transform=None, set_case_id=[], img_size=224, num_repeat=100, stride=1, ar_skip=0, cross_val_train=True, cross_val_fold=None, force_case_sample=False, img_only=False):
        self.split = split

        if split == 'train':
            self.label_dir = os.path.join(data_dir, 'train', 'labels')
            with open('train_meta.json', 'r') as f:
                self.label_meta = json.load(f)
            self.img_dir = os.path.join(data_dir, 'train', 'images')
        else:
            self.label_dir = os.path.join(data_dir, 'val', 'labels')
            with open('val_meta.json', 'r') as f:
                self.label_meta = json.load(f)
            self.img_dir = os.path.join(data_dir, 'val', 'images')

        self.transform = transform

        if len(set_case_id) > 0:
            self.case_list = set_case_id
            logging.info(f'Set Case ID: {set_case_id}')
        else:
            self.case_list = []
            for p in self.label_meta:
                if os.path.isdir(os.path.join(self.label_dir, p)):
                    self.case_list.append(p)
        
        if split == 'train' and cross_val_fold is not None:
            print(f'Use Cross Validation!!! Train={cross_val_train} Fold No. {cross_val_fold}')
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for fold, (train_index, test_index) in enumerate(kf.split(self.case_list)):
                if fold == cross_val_fold:
                    if cross_val_train:
                        self.case_list = [self.case_list[idx] for idx in train_index]
                    else:
                        self.split = split = 'val' # a hack
                        self.case_list = [self.case_list[idx] for idx in test_index]
                    break
        
        self.num_repeat = num_repeat if split == 'train' or force_case_sample else 1
        self.stride = stride
        print(f'Num cases: {len(self.case_list)}, ', 'Num repeat:', self.num_repeat, 'Stride:', self.stride)
        for case_id in self.case_list:
            self.label_meta[case_id]['frames'] = np.array(self.label_meta[case_id]['frames'][::stride])
        

        if split == 'val':
            self.data_list = []
            for case_id in self.case_list:
                frames = self.label_meta[case_id]['frames']
                case_frames = [f'{case_id}/{frame}' for frame in frames]
                self.data_list.extend(case_frames)
            logging.info(f'Eval data list len = {len(self.data_list)}')
            self.data_list = self.data_list * self.num_repeat
            self.data_list = np.asarray(self.data_list)
            
        else:
            self.data_list = None
        

        self.ar_skip = ar_skip # skip the first and last few frames to align with the seq model
        if ar_skip > 0:
            print('AR Skip:', ar_skip)
            
        self.img_size = img_size
        self.mask = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        r = img_size / 256
        cv2.ellipse(self.mask, center=(int(128 * r), int(9 * r)), axes=(int(237*r), int(237*r)), angle=0, startAngle=45,
                            endAngle=135, color=(1, 1, 1), thickness=-1)
        self.mask_image = Image.fromarray((self.mask * 255).astype('uint8')).convert('RGB')
        self.force_case_sample = force_case_sample
        self.img_only = img_only

    def __len__(self) -> int:
        return len(self.case_list) * self.num_repeat if self.split == 'train' or self.force_case_sample else len(self.data_list)

    def read_image(self, img_path):
        image = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        image = np.asarray(image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS))
        image = Image.fromarray(image * self.mask)
        return image

    def __getitem__(self, index):
        if self.split == 'train' or self.force_case_sample:
            index =index % len(self.case_list)
            case_id = self.case_list[index]
            case_meta = self.label_meta[case_id]
            all_files = case_meta['frames']
            # randomly choose an instance
            file = all_files[torch.randint(0, len(all_files), (1,)).item()]
            path = os.path.join(case_id, file)
        else:
            path = self.data_list[index]
            case_id, file = path.split('/')
        
        
        if self.transform is not None:
            img = self.transform(self.read_image(path.replace('.txt', '.jpg')))
        if self.img_only:
            return img
        
        case_meta = self.label_meta[case_id]
        label_path = os.path.join(self.label_dir, path)
        label = open(label_path, 'r').readline()
        if 'nan' in label:
            label = label.replace('nan', str([-0.001, -0.001, -0.001, -1., -1., -1.]))

        
        
        label = torch.tensor(eval(label))
        label = torch.cat([label[:, :3] * 1000, label[:, 3:]], dim=1)
        label = label.float()
        coeff = torch.ones_like(label)
        coeff[(label == -1.).all(dim=1)] = 0
        coeff[(label > 200).any(dim=1) | (label < -200).any(dim=1)] = 0
        
        if self.ar_skip > 0:
            file_idx = case_meta['frames'].index(file)
            if file_idx < self.ar_skip or len(case_meta['frames']) - file_idx <= self.ar_skip:
                coeff = torch.zeros_like(coeff)
            

        return img, label.flatten(), coeff.flatten()




class SeqDatasetYY(txtDatasetYY):
    def __init__(self, seq_len=6, num_plane_to_select=1, ar_eval=False, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        print('Seq len:', seq_len)
        pose_num_count_file_path = f'{self.split}_check_data_available_pose_num.json'
        with open(pose_num_count_file_path, "r") as file:
            pose_num_data = json.load(file)

        idealx_prefix_file_path = f'{self.split}_idealx_image_prefixes.json'
        with open(idealx_prefix_file_path, "r") as file:
            self.idealx_prefix = json.load(file)
        self.idealx_prefix = {k: torch.tensor(v) for k, v in self.idealx_prefix.items()}
        
        outlier_list = []
        exclude_plane = [str(i) for i in range(num_plane_to_select)]
        for key, value in pose_num_data.items():
            if key in exclude_plane:
                outlier_list += value

        for outlier in outlier_list:
            words = outlier.split('/')
            case_id, file = words[-2], words[-1]
            if case_id not in self.label_meta:
                continue
            file = file.replace('.jpg', '.txt')
            if file in self.label_meta[case_id]['frames']:
                self.label_meta[case_id]['frames'] = list(self.label_meta[case_id]['frames'])
                self.label_meta[case_id]['frames'].remove(file)
                self.label_meta[case_id]['frames'] = np.array(self.label_meta[case_id]['frames'])


        self.ar_eval = ar_eval
        if ar_eval:
            print('AR EVAL!!!!!!!!!!!')
        if self.split == 'val':
            self.data_list = []
            for case_id in self.case_list:
                frames = self.label_meta[case_id]['frames']
                case_frames = [f'{case_id}/{frame}' for frame in frames]
                self.data_list.extend(case_frames)
            if ar_eval:
                self.direction = [0] * len(self.data_list) + [1] * len(self.data_list)
                self.data_list = self.data_list + self.data_list

            self.data_list = np.asarray(self.data_list)
        else:
            self.data_list = None


    def _get_label(self, label_path):
        label = open(label_path, 'r').readline()
        if 'nan' in label:
            label = label.replace('nan', str([-0.001, -0.001, -0.001, -1., -1., -1.]))
        label = torch.tensor(eval(label))
        label = torch.cat([label[:, :3] * 1000, label[:, 3:]], dim=1)
        label = label.float()

        ###
        coeff = torch.ones_like(label)
        coeff[(label == -1.).all(dim=1)] = 0
        coeff[(label > 200).any(dim=1) | (label < -200).any(dim=1)] = 0

        ### Outliers set as 360
        mask = (label > 200).any(dim=1) | (label < -200).any(dim=1)
        label[mask] = 360
        mask = (label == -1.).all(dim=1)
        label[mask] = 360

        return label, coeff

    def __getitem__(self, index):
        if self.split == 'train':
            index =index % len(self.case_list)
            case_id = self.case_list[index]
            case_meta = self.label_meta[case_id]
            all_files = case_meta['frames']
            # randomly choose an instance
            file = all_files[torch.randint(0, len(all_files), (1,)).item()]
            path = os.path.join(case_id, file)
        else:
            path = self.data_list[index]
            case_id, file = path.split('/')
        
        case_frames = self.label_meta[case_id]['frames']
        
        img_source_label, coeff = self._get_label(os.path.join(self.label_dir, path))

        # img_seq = [self.read_image(path.replace('.txt', '.jpg'))]
        img_seq_label = [img_source_label]
        selected_file = [file]
        img_seq_name_id = [int(file[:6])]
        act_seq = []
        cur_possible_list = deepcopy(case_frames)
        
        if self.ar_eval:
            ideal_indices = torch.tensor(self.idealx_prefix[case_id])
            this_idx = int(file[:6])
            if self.direction[index] == 0:
                # left to right
                new_cur_possible_list = cur_possible_list[:cur_possible_list.index(file)]
                ideals_to_skip = torch.logical_or(ideal_indices == 1000000, ideal_indices < this_idx)
            else:
                # right to left
                new_cur_possible_list = cur_possible_list[cur_possible_list.index(file)+1:]
                ideals_to_skip = torch.logical_or(ideal_indices == 1000000, ideal_indices >= this_idx)
            ar_valid = len(new_cur_possible_list) >= self.seq_len - 1
            
            # to align with ar_skip in txt dataset
            file_idx = case_frames.index(file)
            idx_valid = not(file_idx < self.seq_len - 1 or len(case_frames) - file_idx <= self.seq_len - 1)
            
            if ar_valid and idx_valid:
                cur_possible_list = new_cur_possible_list
                coeff[ideals_to_skip] = 0
            else:
                coeff = torch.zeros_like(coeff)
                cur_possible_list.remove(file)
        else:
            cur_possible_list.remove(file)

        for _ in range(self.seq_len - 1):
            cur_target_file = random.choice(cur_possible_list)
            cur_possible_list.remove(cur_target_file)
            
            cur_target_name_id = int(cur_target_file[:6])
            cur_target_label, _ = self._get_label(os.path.join(self.label_dir, case_id, cur_target_file))

            # -- calculate distance to ideal plane
            distance = torch.abs(torch.tensor(self.idealx_prefix[case_id]) - img_seq_name_id[-1]) + torch.abs(
                torch.tensor(self.idealx_prefix[case_id]) - cur_target_name_id)
            distance_sorted_indices = torch.argsort(distance)

            # -- calculate relative pose between source and target image
            mask1 = (img_seq_label[-1] == 360).all(dim=1)
            mask2 = (cur_target_label == 360).all(dim=1)
            indices = torch.nonzero(~mask1 & ~mask2).squeeze(1)
            for num, item in enumerate(distance_sorted_indices):
                if item in indices:
                    selected_indices = item
                    break
                    # if num >0:
                    #     print(num)
                    #     print('distance_sorted_indices', distance_sorted_indices)
                    #     print('indices', indices)
            cur_action = Transformation.hexa_diff_inv(img_seq_label[-1][selected_indices].squeeze(), cur_target_label[selected_indices].squeeze())

            # -- update
            selected_file.append(cur_target_file)
            img_seq_label.append(cur_target_label)
            img_seq_name_id.append(cur_target_name_id)
            act_seq.append(cur_action)
        
        img_seq = [self.transform(self.read_image(os.path.join(self.img_dir, case_id, f.replace('.txt', '.jpg')))) for f in selected_file]
        img_seq = torch.stack(img_seq, dim=0)
        if self.seq_len > 1:
            act_seq = torch.from_numpy(np.stack(act_seq, axis=0).astype(np.float32))
            return img_seq, act_seq, img_source_label.flatten(), coeff.flatten()
        else:
            return img_seq, img_source_label.flatten(), coeff.flatten()




class PairDatasetYY(SeqDatasetYY):
    def __init__(self, weight_mode='', joint=False, with_inv=False, **kwargs):
        super().__init__(**kwargs)
        if self.split == 'train':
            with open('train_meta_1026.json', 'r') as f:
                self.label_meta = json.load(f) # remove all null frames
        self.frame_weights = None
        self.joint = joint
        self.with_inv = with_inv
        
    
    def __getitem__(self, index):
        if self.split == 'train':
            index =index % len(self.case_list)
            case_id = self.case_list[index]
            case_meta = self.label_meta[case_id]
            all_files = case_meta['frames']
            # randomly choose an instance
            if self.frame_weights is not None:
                file = all_files[torch.multinomial(self.frame_weights[case_id], num_samples=1).item()]
            else:
                file = all_files[torch.randint(0, len(all_files), (1,)).item()]

            path = os.path.join(case_id, file)
        else:
            path = self.data_list[index]
            case_id, file = path.split('/')
        
        case_frames = self.label_meta[case_id]['frames']
        
        img_source_label, coeff1 = self._get_label(os.path.join(self.label_dir, path))

        img_source_id = int(file[:6])
        
        if self.frame_weights is not None:
            cur_target_file = case_frames[torch.multinomial(self.frame_weights[case_id], num_samples=1).item()]
        else:
            cur_target_file= case_frames[torch.randint(0, len(case_frames), (1,)).item()]
        
        cur_target_name_id = int(cur_target_file[:6])
        
        cur_target_label, coeff2 = self._get_label(os.path.join(self.label_dir, case_id, cur_target_file))

        # -- calculate distance to ideal plane
        distance = torch.abs(self.idealx_prefix[case_id] - img_source_id) + torch.abs(
            self.idealx_prefix[case_id] - cur_target_name_id)
        distance_sorted_indices = torch.argsort(distance)

        # -- calculate relative pose between source and target image
        mask1 = (img_source_label == 360).all(dim=1)
        mask2 = (cur_target_label == 360).all(dim=1)
        indices = torch.nonzero(~mask1 & ~mask2).squeeze(1)
        # for num, item in enumerate(distance_sorted_indices):
        #     if item in indices:
        #         selected_indices = item
        #         break
        selected_indices = next((item for item in distance_sorted_indices if item in indices), None)
        cur_action = Transformation.hexa_diff_inv(img_source_label[selected_indices].squeeze(), cur_target_label[selected_indices].squeeze())
        cur_action_inv = Transformation.hexa_diff_inv(cur_target_label[selected_indices].squeeze(), img_source_label[selected_indices].squeeze(), )
        
        img_source = self.read_image(os.path.join(self.img_dir, case_id, file.replace('.txt', '.jpg')))
        img_target = self.read_image(os.path.join(self.img_dir, case_id, cur_target_file.replace('.txt', '.jpg')))
        if self.with_inv:
            images = torch.stack([self.transform(img_source), self.transform(img_source), self.transform(img_target), self.transform(img_target)], dim=0)
            return images, cur_action, cur_action_inv, img_source_label.flatten(), cur_target_label.flatten(), coeff1.flatten(), coeff2.flatten()
        else:
            img_source, img_target = self.transform(img_source), self.transform(img_target)
            if self.joint:
                return img_source, img_target, cur_action, cur_action_inv, img_source_label.flatten(), cur_target_label.flatten(), coeff1.flatten(), coeff2.flatten()
            else:
                return img_source, img_target, cur_action, cur_action_inv




class CaseRepeatWrapper(Dataset):
    def __init__(self, base_dataset: PairDatasetYY, case_num_repeat=32, num_repeat=1):
        super().__init__()
        self.base_dataset = base_dataset
        self.case_num_repeat = case_num_repeat
        self.num_repeat = num_repeat

    
    def __len__(self):
        return len(self.base_dataset.case_list * self.num_repeat)

    def __getitem__(self, idx):
        idx = idx % len(self.base_dataset.case_list)
        data = [self.base_dataset[idx] for _ in range(self.case_num_repeat)]
        stacked_data = [torch.stack([torch.tensor(d[item_idx]) for d in data], dim=0) for item_idx in range(len(data[0]))]    
        return stacked_data



# class DiffDatasetYY(Dataset):
#     def __init__(self, data_dir='/home/yueyang/ultrasound_labels', img_dir='/data/yueyang/ultrasound_images', train=True, transform=None, data_pct=1.0, imgs_per_case=8):
#         if train:
#             self.label_dir = os.path.join(data_dir, 'train', 'labels')
#             with open(os.path.join(data_dir, 'train_meta.json'), 'r') as f:
#                 self.label_meta = json.load(f)
#         else:
#             self.label_dir = os.path.join(data_dir, 'val', 'labels')
#             with open(os.path.join(data_dir, 'val_meta.json'), 'r') as f:
#                 self.label_meta = json.load(f)
#         self.train = train
#         self.img_dir = os.path.join(img_dir, 'images')
#         self.transform = transform

#         self.case_list = []
#         for p in os.listdir(self.label_dir):
#             if os.path.isdir(os.path.join(self.label_dir, p)):
#                 self.case_list.append(p)
#         if data_pct < 1.0:
#             _, self.case_list  = train_test_split(self.case_list , test_size=data_pct, random_state=0)
#         self.imgs_per_case = imgs_per_case
#         print(f'Num Cases: {len(self.case_list)}')

#     def __len__(self) -> int:
#         return len(self.case_list)


#     def _get_label(self, label_path):
#         label = open(label_path, 'r').readline()
#         if 'nan' in label:
#             label = label.replace('nan', str([-0.001, -0.001, -0.001, -1., -1., -1.]))
#         label = torch.tensor(eval(label))
#         label = torch.cat([label[:, :3] * 1000, label[:, 3:]], dim=1)
#         label = label.float()

#         ###
#         coeff = torch.ones_like(label)
#         coeff[(label == -1.).all(dim=1)] = 0
#         coeff[(label > 200).any(dim=1) | (label < -200).any(dim=1)] = 0

#         ### Outliers set as 360
#         mask = (label > 200).any(dim=1) | (label < -200).any(dim=1)
#         label[mask] = 360
#         mask = (label == -1.).all(dim=1)
#         label[mask] = 360

#         return label, coeff

#     def _get_diff(self, label1, label2):
#         mask1 = (label1 == 360).all(dim=1)
#         mask2 = (label2 == 360).all(dim=1)
#         indices = torch.nonzero(~mask1 & ~mask2).squeeze(1).tolist()
#         if len(indices) == 0:
#             return None # no available indices
#         else:
#             selected_indices = random.choice(indices)
#             diff = Transformation.hexa_diff_inv(label1[selected_indices].squeeze(),
#                                                     label2[selected_indices].squeeze())
#             return diff
    
#     def __getitem__(self, index):
#         case_id = self.case_list[index]
#         case_meta = self.label_meta[case_id]
#         all_frames = case_meta['frames']
#         if self.train:
#             sampled_frames = random.sample(all_frames, self.imgs_per_case)
#         else:
#             sampled_indices = np.linspace(0, len(all_frames), self.imgs_per_case+1)
#             sampled_indices = [int((i+j)/2) for i,j in zip(sampled_indices[:-1], sampled_indices[1:])]
#             sampled_frames = [all_frames[i] for i in sampled_indices]
#         imgs = [Image.open(os.path.join(self.img_dir, case_id, frame.replace('.txt', '.jpg'))).convert('RGB').resize((224, 224)) for frame in sampled_frames]
#         imgs = [self.transform(img) for img in imgs]
#         imgs = torch.stack(imgs, dim=0)
#         labels = [self._get_label(os.path.join(self.label_dir, case_id, frame))[0] for frame in sampled_frames]
#         targets = np.zeros((self.imgs_per_case, self.imgs_per_case, 6), dtype=float)
#         targets_valid = np.ones((self.imgs_per_case, self.imgs_per_case), dtype=float)
#         for idx1, label1 in enumerate(labels):
#             for idx2, label2 in enumerate(labels):
#                 diff = self._get_diff(label1, label2)
#                 if diff is None:
#                     targets_valid[idx1, idx2] = 0
#                 else:
#                     targets[idx1, idx2] = diff
        
#         return  imgs, targets, targets_valid




