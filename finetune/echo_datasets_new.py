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

class txtDatasetYY(Dataset):
    def __init__(self, split='train', data_dir='yueyang/modified_dataset', transform=None, set_case_id=[], img_size=224, num_repeat=100, stride=1, ar_skip=0, cross_val_train=True, cross_val_fold=None, nfolds=5, force_sampling=None, img_only=False, data_pct=1.0):
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

        if split == 'train' and data_pct < 1.0:
            self.case_list, _ = train_test_split(self.case_list, test_size=(1-data_pct), random_state=42)
        pose_num_count_file_path = f'{self.split}_check_data_available_pose_num.json'
        with open(pose_num_count_file_path, "r") as file:
            pose_num_data = json.load(file)

        idealx_prefix_file_path = f'{self.split}_idealx_image_prefixes.json'
        with open(idealx_prefix_file_path, "r") as file:
            self.idealx_prefix = json.load(file)
        self.idealx_prefix = {k: torch.tensor(v) for k, v in self.idealx_prefix.items()}
        
        outlier_list = []
        exclude_plane = [str(i) for i in range(1)]
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
                self.label_meta[case_id]['frames'].remove(file)
        
        
        if split == 'train' and cross_val_fold is not None:
            print(f'Use Cross Validation!!! Train={cross_val_train} Fold No. {cross_val_fold}')
            kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)
            for fold, (train_index, test_index) in enumerate(kf.split(self.case_list)):
                if fold == cross_val_fold:
                    if cross_val_train:
                        self.case_list = [self.case_list[idx] for idx in train_index]
                    else:
                        self.split = split = 'val' # a hack
                        self.case_list = [self.case_list[idx] for idx in test_index]
                    break

        self.sampling_method = 'case' if split == 'train' else 'frame'
        if force_sampling:
            self.sampling_method = force_sampling
        self.num_repeat = num_repeat
        self.stride = stride
        print(f'Num cases: {len(self.case_list)}, ', 'Num repeat:', self.num_repeat, 'Stride:', self.stride)
        
        
        for case_id in self.case_list:
            # self.label_meta[case_id]['frames'] = np.array(self.label_meta[case_id]['frames'][::stride])
            self.label_meta[case_id]['frames'] = self.label_meta[case_id]['frames'][::stride]
        
        self.case_start_end = {}
        self.data_list = []
        for case_id in self.case_list:
            frames = self.label_meta[case_id]['frames']
            case_frames = [f'{case_id}/{frame}' for frame in frames]
            self.case_start_end[case_id] = (len(self.data_list), len(self.data_list)+len(frames))
            self.data_list.extend(case_frames)
        logging.info(f'data list len = {len(self.data_list)}')
        
        self.data_list = np.asarray(self.data_list)
        self.img_size = img_size
        self.mask = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        r = img_size / 256
        cv2.ellipse(self.mask, center=(int(128 * r), int(9 * r)), axes=(int(237*r), int(237*r)), angle=0, startAngle=45,
                            endAngle=135, color=(1, 1, 1), thickness=-1)
        self.mask_image = Image.fromarray((self.mask * 255).astype('uint8')).convert('RGB')
        self.img_only = img_only

    def __len__(self) -> int:
        if self.sampling_method == 'case':
            return len(self.case_list) * self.num_repeat
        else:
            return len(self.data_list)

    def read_image(self, img_path):
        image = Image.open(os.path.join(self.img_dir, img_path)).convert('RGB')
        image = np.asarray(image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS))
        image = Image.fromarray(image * self.mask)
        return image


    def _get_label(self, label_path):
        with open(label_path, 'r') as f:
            label = f.readline()
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
    
    def _get_diff(self, label1, label2, id1, id2, case_id):
        # -- calculate distance to ideal plane
        distance = torch.abs(torch.tensor(self.idealx_prefix[case_id]) - id1) + torch.abs(
            torch.tensor(self.idealx_prefix[case_id]) - id2)
        distance_sorted_indices = torch.argsort(distance)

        # -- calculate relative pose between source and target image
        mask1 = (label1 == 360).all(dim=1)
        mask2 = (label2 == 360).all(dim=1)
        indices = torch.nonzero(~mask1 & ~mask2).squeeze(1)
        for num, item in enumerate(distance_sorted_indices):
            if item in indices:
                selected_indices = item
                break
                # if num >0:
                #     print(num)
                #     print('distance_sorted_indices', distance_sorted_indices)
                #     print('indices', indices)
        return Transformation.hexa_diff_inv(label1[selected_indices].squeeze(), label2[selected_indices].squeeze())

    def __getitem__(self, index):
        if self.sampling_method == 'case':
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
        
        label_path = os.path.join(self.label_dir, path)
        label, coeff = self._get_label(label_path)
        return img, label.flatten(), coeff.flatten()



class MultiFrameDatasetPolicy(txtDatasetYY):
    def __init__(self, num_frames=8, ar_eval=False, ar_train=False, sample_mode='exp', exp_alpha=0.25, pred_every=False, act_mode='full', seq_prob=1.0, eval_indices=None, eval_indices_plane=None, **kwargs):
        '''
        
        '''
        super().__init__(**kwargs)
        self.num_frames = num_frames
        self.pred_every = pred_every
        self.act_mode = act_mode
        self.sample_mode = sample_mode
        self.exp_alpha = exp_alpha
        self.seq_prob = seq_prob
        self.eval_indices = eval_indices
        self.eval_indices_plane = eval_indices_plane
        if sample_mode == 'exp':
            logging.info(f'Alpha={exp_alpha}, example sample: {self.sample_timesteps_exponential(100, num_frames)}')
        
        self.ar_eval = ar_eval
        self.ar_train = ar_train
        if ar_train:
            logging.info('AR Train!!!')
        if ar_eval:
            self.direction = [0] * len(self.data_list) + [1] * len(self.data_list)
        
    def __len__(self):
        if self.eval_indices_plane is not None:
            return len(self.data_list) * 2 * 10
        elif self.ar_eval or self.eval_indices is not None:
            return len(self.data_list) * 2
        else:
            return super().__len__()
    
    
    def sample_timesteps_exponential(self, t, K):
        # Calculate the total duration
        t_min = 0
        alpha = self.exp_alpha
        D = t - t_min + 1
        # Set the rate parameter lambda with scaling factor
        lambd = alpha * K / D
        # Compute quantiles centered between 0 and 1
        q = (np.arange(1, K + 1) - 0.5) / K
        # Calculate time intervals using the inverse exponential CDF
        tau = - (1 / lambd) * np.log(1 - q)
        # Determine sampled timesteps
        t_samples = t - tau
        # Ensure sampled timesteps do not go beyond t_min
        t_samples = np.maximum(t_samples, t_min)
        return np.round(t_samples).astype(int)[::-1]

    
    # def _get_diff_simple_grid(self, labels):
    #     B = labels.shape[0]
    #     mask = (labels == 360).all(dim=2)
    #     combined_mask = ~mask.any(dim=0)
    #     indices = torch.nonzero(combined_mask).squeeze(1).tolist()
    #     selected_indices = random.choice(indices)
    #     labels_selected = labels[:, selected_indices] #[B, 6]
        
    #     matrices = np.stack([Transformation.hexa2trans(l, order='xyz', degrees=True) for l in labels_selected], axis=0) #[B, 4, 4]
    #     matrices = torch.from_numpy(matrices)
    #     matrices_inv = torch.inverse(matrices)
    #     M1 = matrices[:, None, :, :]          # Shape: (B, 1, 4, 4)
    #     M2_inv = matrices_inv[None, :, :, :]  # Shape: (1, B, 4, 4)
        
    #     matrices_diff = (M1 @ M2_inv).reshape(-1, 4, 4).numpy()
    #     hexa_diff = np.stack([Transformation.trans2hexa(t, order='xyz', degrees=True) for t in matrices_diff], axis=0)

    #     return hexa_diff.reshape(B, B, 6)     
        
    def _get_diff_simple(self, label1, label2):
        mask1 = (label1 == 360).all(dim=1)
        mask2 = (label2 == 360).all(dim=1)
        indices = torch.nonzero(~mask1 & ~mask2).squeeze(1).tolist()
        if len(indices) == 0:
            return None # no available indices
        else:
            selected_indices = random.choice(indices)
            diff = Transformation.hexa_diff_inv(label1[selected_indices].squeeze(),
                                                    label2[selected_indices].squeeze())
            return diff
    
    def getitem_ar(self, index, case_frames, ar_direction, sampling_mode='exp'):
        # current frame is the last frame
        if self.num_frames == 1:
            selected_indices = [index]
        elif sampling_mode == 'random':
            if ar_direction == 0:
                selected_indices = random.choices(np.arange(index+1), k=self.num_frames-1)
                selected_indices.sort()
            else:
                selected_indices = random.choices(np.arange(index, len(case_frames)), k=self.num_frames-1)
                selected_indices.sort()
                selected_indices = selected_indices[::-1]
            selected_indices.append(index)
        else:
            if ar_direction == 0:
                # left to right
                selected_indices = self.sample_timesteps_exponential(index, self.num_frames - 1)
            else:
                selected_indices = self.sample_timesteps_exponential(len(case_frames)-index-1, self.num_frames - 1)
                selected_indices = len(case_frames) - 1 - selected_indices
            selected_indices = selected_indices.tolist() + [index]

        assert len(selected_indices) == self.num_frames
        selected_frames = [case_frames[idx] for idx in selected_indices]
        return selected_frames
    
    def __getitem__(self, index):
        if self.eval_indices_plane is not None:
            B, C = 2, 10
            sample_idx = index // (B * C)
            rem = index % (B * C)
            direction = rem // C
            plane = rem % C
            path = self.data_list[sample_idx]
            case_id  = path.split('/')[0]
            selected_indices = self.eval_indices_plane[sample_idx, direction, plane]
            selected_frames = [self.label_meta[case_id]['frames'][int(sidx)] for sidx in selected_indices]
        elif self.eval_indices is not None:
            sample_idx = index % len(self.data_list)
            path = self.data_list[sample_idx]
            direction = index // len(self.data_list)
            case_id  = path.split('/')[0]
            selected_indices = self.eval_indices[sample_idx, direction]
            selected_frames = [self.label_meta[case_id]['frames'][int(sidx)] for sidx in selected_indices]
        elif self.ar_eval:
            path = self.data_list[index % len(self.data_list)]
            direction = self.direction[index]
            case_id, filename = path.split('/')
            case_frames = self.label_meta[case_id]['frames']
            selected_frames = self.getitem_ar(case_frames.index(filename), case_frames, direction, self.sampling_method)
        else:
            if self.sampling_method == 'frame':
                path = self.data_list[index % len(self.data_list)]
                case_id, filename = path.split('/')
                case_frames = self.label_meta[case_id]['frames']
                direction = 0 # dummy, not used
                selected_frames = self.getitem_ar(case_frames.index(filename), case_frames, direction, self.sampling_method)
            else:
                index = index % len(self.case_list)
                case_id = self.case_list[index]
                case_frames = self.label_meta[case_id]['frames']
                
                pivot_idx = torch.randint(0, len(case_frames), (1,)).item()
                direction = torch.randint(0, 2, (1,)).item()
                
                mode = 'exp' if torch.rand(1) < self.seq_prob else 'random'
                selected_frames = self.getitem_ar(pivot_idx, case_frames, ar_direction=direction, sampling_mode=mode)


        labels = [self._get_label(os.path.join(self.label_dir, case_id, frame)) for frame in selected_frames]
        labels, labels_coeff = [l[0] for l in labels], [l[1] for l in labels]
        labels = torch.stack(labels, dim=0)
        labels_coeff = torch.stack(labels_coeff, dim=0)
        
        imgs = [self.transform(self.read_image(os.path.join(self.img_dir, case_id, f.replace('.txt', '.jpg')))) for f in selected_frames]
        imgs = torch.stack(imgs, dim=0) #[N,C,H,W]
        
        if self.act_mode == 'full':
            rel_pos = np.zeros((self.num_frames, self.num_frames, 6), dtype=float)
            for idx1, label1 in enumerate(labels):
                for idx2, label2 in enumerate(labels):
                    diff = self._get_diff_simple(label1, label2)
                    assert diff is not None
                    rel_pos[idx1, idx2] = diff
            # rel_pos =self._get_diff_simple_grid(labels)
            res = [imgs, rel_pos]
        elif self.act_mode == 'seq_inv':
            # sequential reversed, predict the first one
            acts = np.zeros((self.num_frames-1, 6), dtype=float)
            for fidx in range(self.num_frames-1):
                acts[fidx] = self._get_diff_simple(labels.flip(0)[fidx], labels.flip(0)[fidx+1])
            imgs = imgs.flip(0)
            res = [imgs, acts]
        elif self.act_mode == 'seq':
            # sequential, only predict the last one
            acts = np.zeros((self.num_frames-1, 6), dtype=float)
            for fidx in range(self.num_frames-1):
                acts[fidx] = self._get_diff_simple(labels[fidx], labels[fidx+1])
            res = [imgs, acts]
        else:
            raise ValueError
    
        # in case we do ar training or eval
        ideal_indices = self.idealx_prefix[case_id]
        this_idx = int(selected_frames[-1][:6])
        if direction == 0:
            # left to right
            ideals_to_skip = torch.logical_or(ideal_indices == 1000000, ideal_indices < this_idx)
        else:
            ideals_to_skip = torch.logical_or(ideal_indices == 1000000, ideal_indices >= this_idx)
        
        ### process labels
        if self.ar_train:
            labels_coeff = labels_coeff.view(-1, 10, 6)
            labels_coeff[:, ideals_to_skip] = 0
            res.extend([labels.flatten(1), labels_coeff.flatten(1)])
        elif self.ar_eval or self.eval_indices is not None or self.eval_indices_plane is not None: # ar inference only
            rel_pos_to_others = np.zeros((self.num_frames, 6), dtype=float)
            for fidx in range(self.num_frames):
                rel_pos_to_others[fidx] = self._get_diff_simple(labels[-1], labels[fidx])
             # use the last one
            labels_coeff = labels_coeff[-1].view(10, 6)
            labels_coeff[ideals_to_skip] = 0
            if self.eval_indices_plane is not None:
                # only predict current plane
                labels_coeff[np.arange(10) != plane] = 0
            labels_coeff = labels_coeff.flatten()
            res.extend([labels[-1].flatten(), labels_coeff, rel_pos_to_others])
        elif self.pred_every:
            res.extend([labels.flatten(1), labels_coeff.flatten(1)])
        else:
            res.extend([labels[-1].flatten(), labels_coeff[-1].flatten()]) 

        return res


