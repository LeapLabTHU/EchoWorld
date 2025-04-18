# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
# try:
#     # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
#     # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
#     # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
#     # --          TO EACH PROCESS
#     os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
# except Exception:
#     pass

import copy
import logging
import sys
import yaml
import argparse
import random
import pprint
import warnings
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel
from src.models.vision_transformer import trunc_normal_
from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.echo_datasets import make_echo_dataset

from src.helper import (
    load_checkpoint,
    init_model,
    # load_checkpoint_cardiac_v3,
    # init_cardiac_model_v4,
    init_opt)
# from src.transforms import make_cardiac_seq_transforms
from src.transforms import make_transforms
# --

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def info_nce_loss(features,batch_size=None, temp=0.1):
    if batch_size is None:
        batch_size = features.shape[0] // 2
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

    logits = logits / temp
    return F.cross_entropy(logits, labels)



def compute_recall_at_k(h_hat, h, ks=[1, 5, 10], gather=False):
    
    # Ensure h_hat and h are PyTorch tensors
    if not isinstance(h_hat, torch.Tensor):
        h_hat = torch.tensor(h_hat)
    if not isinstance(h, torch.Tensor):
        h = torch.tensor(h)
    if gather:
        h = torch.cat(FullGatherLayer.apply(h), dim=0)
        h_hat = torch.cat(FullGatherLayer.apply(h_hat), dim=0)
    
    N, D = h.shape
    # Compute pairwise L2 distances between predicted and ground truth features
    # distances: N x N tensor
    distances = torch.cdist(h_hat, h, p=2)  # Shape: N x N

    # Get the indices that sort the distances for each predicted feature
    sorted_indices = torch.argsort(distances, dim=1)  # Shape: N x N

    recalls = {}
    for k in ks:
        # Get the top-k indices for each predicted feature
        top_k_indices = sorted_indices[:, :k]  # Shape: N x k

        # Ground truth indices (from 0 to N-1)
        ground_truth_indices = torch.arange(N, device=h_hat.device).unsqueeze(1)  # Shape: N x 1

        # Check if the ground truth index is among the top-k predictions
        correct = (top_k_indices == ground_truth_indices).any(dim=1).float()  # Shape: N

        # Calculate recall@k
        recalls[k] = correct.mean().item()
    return recalls

log_timings = True
log_freq = 50
checkpoint_freq = 20
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str,
                    help='name of config file to load', default='configs.yaml')
# parser.add_argument(
#     '--devices', type=str, nargs='+', default=['cuda:0'],
#     help='which devices to use on local machine')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
parser.add_argument('--extra_mode', type=str, default='simclr')
parser.add_argument('--intra_weight', type=float, default=1.0)
parser.add_argument('--extra_weight', type=float, default=0.1)
parser.add_argument('--detach_extra', action='store_true', default=False)
parser.add_argument('--output', type=str, default=None)

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn(
                "nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'")
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args, resume_preempt=False):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        # device = torch.device('cuda:0')
        device = args.gpu
        torch.cuda.set_device(device)

    rank, world_size = args.rank, args.world_size
    fname = args.fname

    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        print('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # -- LOGGING
    folder = params['logging']['folder']
    if args.output is not None:
        folder = args.output
    if not os.path.exists(folder):
        os.makedirs(folder)
    tag = params['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(params, f)
    # ----------------------------------------------------------------------- #

    logfile_path = os.path.join(params['logging']['folder'], 'exp.log')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(logfile_path, mode='a'),  # 日志写入文件
                            logging.StreamHandler()  # 日志输出到控制台
                        ])
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = params['meta']['use_bfloat16']
    model_name = params['meta']['model_name']
    # assert model_name == 'vit_small_v4'
    load_model = params['meta']['load_checkpoint'] or resume_preempt
    r_file = params['meta']['read_checkpoint']
    copy_data = params['meta']['copy_data']
    pred_depth = params['meta']['pred_depth']
    pred_emb_dim = params['meta']['pred_emb_dim']
    pred_num_heads = params['meta']['pred_num_heads']
    pred_mlp_ratio = params['meta']['pred_mlp_ratio']

    # -- DATA
    use_gaussian_blur = params['data']['use_gaussian_blur']
    use_horizontal_flip = params['data']['use_horizontal_flip']
    use_color_distortion = params['data']['use_color_distortion']
    color_jitter = params['data']['color_jitter_strength']
    # --
    batch_size = params['data']['batch_size']
    pin_mem = params['data']['pin_mem']
    num_workers = params['data']['num_workers']
    root_path = params['data']['root_path']
    image_folder = params['data']['image_folder']
    crop_size = params['data']['crop_size']
    crop_scale = params['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = params['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = params['mask']['patch_size']  # patch-size for model training
    num_enc_masks = params['mask']['num_enc_masks']  # number of context blocks
    min_keep = params['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = params['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = params['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = params['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = params['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    paper_original_total_batchsize = 2048
    current_total_batchsize = batch_size * world_size
    lr_modify_ratio = current_total_batchsize / paper_original_total_batchsize
    logger.info(
        f'Paper original/current total batchsize, lr modify ratio: {paper_original_total_batchsize}/{current_total_batchsize}, {lr_modify_ratio}')

    ema = params['optimization']['ema']
    ipe_scale = params['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(params['optimization']['weight_decay'])
    final_wd = float(params['optimization']['final_weight_decay'])
    num_epochs = params['optimization']['epochs']
    warmup = params['optimization']['warmup']
    start_lr = params['optimization']['start_lr'] * lr_modify_ratio
    lr = params['optimization']['lr'] * lr_modify_ratio
    final_lr = params['optimization']['final_lr'] * lr_modify_ratio

    logger.info(f'Current start_lr/lr/final_lr: {start_lr}/{lr}/{final_lr}')

    # -- LOGGING
    folder = params['logging']['folder']
    if not os.path.exists(folder):
        os.makedirs(folder)
    tag = params['logging']['write_tag']

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(params, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # # -- init torch distributed backend
    # world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    # csv_logger = CSVLogger(log_file,
    #                        ('%d', 'epoch'),
    #                        ('%d', 'itr'),
    #                        ('%.5f', 'loss'),
    #                        ('%.5f', 'loss-img'),
    #                        ('%.5f', 'loss-act'),
    #                        ('%.5f', 'mask-A'),
    #                        ('%.5f', 'mask-B'),
    #                        ('%d', 'time (ms)'))
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))
    # -- init model
    # encoder, predictor = init_cardiac_model_v4(
    #     device=device,
    #     patch_size=patch_size,
    #     crop_size=crop_size,
    #     pred_depth=pred_depth,
    #     pred_emb_dim=pred_emb_dim,
    #     pred_num_heads=pred_num_heads,
    #     pred_mlp_ratio=pred_mlp_ratio,
    #     model_name=model_name,
    #     timestep=timestep)
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    if args.extra_mode == 'simclr':
        encoder.simclr_proj = nn.Sequential(
                    nn.Linear(384, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, 1024, bias=False)
                ).cuda()
    target_encoder = copy.deepcopy(encoder)

    predictor.policy_net = nn.Sequential(
            nn.Linear(6, pred_emb_dim),
            nn.SiLU(),
            nn.Linear(pred_emb_dim, pred_emb_dim),
        ).cuda()
    predictor.mask_token_extra = nn.Parameter(torch.zeros(1, 1, pred_emb_dim)).cuda()
    trunc_normal_(predictor.mask_token_extra, std=predictor.init_std)
    
    # -- make data transforms
    # mask_collator = CardiacMaskCollatorV4(
    #     input_size=crop_size,
    #     patch_size=patch_size,
    #     pred_mask_scale=pred_mask_scale,
    #     enc_mask_scale=enc_mask_scale,
    #     aspect_ratio=aspect_ratio,
    #     nenc=num_enc_masks,
    #     npred=num_pred_masks,
    #     allow_overlap=allow_overlap,
    #     min_keep=min_keep,
    #     timestep=timestep
    # )
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)
    # transform = make_cardiac_seq_transforms(
    #     crop_size=crop_size,
    #     crop_scale=crop_scale,
    #     gaussian_blur=use_gaussian_blur,
    #     horizontal_flip=use_horizontal_flip,
    #     color_distortion=use_color_distortion,
    #     color_jitter=color_jitter)
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)
    # -- init data-loaders/samplers
    # _, unsupervised_loader, unsupervised_sampler = make_cardiac_seq_dataset_v3(
    #     transform=transform,
    #     batch_size=batch_size,
    #     collator=mask_collator,
    #     pin_mem=pin_mem,
    #     training=True,
    #     num_workers=num_workers,
    #     world_size=world_size,
    #     rank=rank,
    #     root_path=root_path,
    #     image_folder=image_folder,
    #     copy_data=copy_data,
    #     drop_last=True,
    #     num_plane_to_select=num_plane_to_select,
    #     timestep=timestep)
    _, unsupervised_loader, unsupervised_sampler = make_echo_dataset(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True, pair=True)
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    encoder = DistributedDataParallel(encoder)
    predictor = DistributedDataParallel(predictor)
    target_encoder = DistributedDataParallel(target_encoder)

    for p in target_encoder.parameters():
        p.requires_grad = False
    # for p in encoder.parameters():
    #     p.requires_grad = False

    total_params = sum(p.numel() for p in encoder.parameters()) / 1e6
    print('Total number of [Encoder] parameters: {} M'.format(total_params))
    total_params = sum(p.numel() for p in predictor.parameters()) / 1e6
    print('Total number of [Predictor] parameters: {} M'.format(total_params))

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        # print('Loading pretrained I-JEPA model...')
        # encoder, target_encoder = load_checkpoint_cardiac_v3(
        #     r_path=load_path,
        #     encoder=encoder,
        #     target_encoder=target_encoder)
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
                device=device,
                r_path=load_path,
                encoder=encoder,
                predictor=predictor,
                target_encoder=target_encoder,
                opt=optimizer,
                scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch+1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=epoch))

    # -- TRAINING LOOP
    logger.info('Dataset size %d' % len(unsupervised_loader))
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        loss_intra_meter = AverageMeter()
        loss_extra_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        recall1_meter = AverageMeter()
        recall5_meter = AverageMeter()
        recall10_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, data in enumerate(unsupervised_loader):

            def nested_to_gpu(nested_list, device='cuda'):
                if isinstance(nested_list, torch.Tensor):
                    return nested_list.to(device, non_blocking=True)  # Move tensor to GPU
                else:
                    return [nested_to_gpu(item, device) for item in nested_list]  # Recursively process list
            (image1, image2, rel_pos12, rel_pos21), masks_enc, masks_pred = nested_to_gpu(data)
            
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target(imgs):
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h_mask = apply_masks(h, masks_pred)
                        h_mask = repeat_interleave_batch(h_mask, B, repeat=len(masks_enc))
                        return h_mask, h.mean(1)

                def forward_context(z):
                    z = predictor(z, masks_enc, masks_pred)
                    return z
                
                def forward_context_extra(z, acts):
                    if args.detach_extra:
                        z = z.detach()
                    z = predictor(z, masks_enc, acts, extra=True)
                    return z.mean(1)

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h1, h1_mean = forward_target(image1)
                    h2, h2_mean = forward_target(image2)
                    z1 = encoder(image1, masks_enc)
                    z2 = encoder(image2, masks_enc)
                    
                    z11 = forward_context(z1)
                    z22 = forward_context(z2)
                    z12 = forward_context_extra(z1, rel_pos12.float() / 200) # proc label
                    z21 = forward_context_extra(z2, rel_pos21.float() / 200)
                    
                    loss_intra = (loss_fn(z11, h1) + loss_fn(z22, h2)) / 2
                    
                    if args.extra_mode == 'simclr':
                        h1_mean_g = torch.cat(FullGatherLayer.apply(target_encoder.module.simclr_proj(h1_mean)), dim=0)
                        h2_mean_g = torch.cat(FullGatherLayer.apply(target_encoder.module.simclr_proj(h2_mean)), dim=0)
                        z12_g = torch.cat(FullGatherLayer.apply(encoder.module.simclr_proj(z12)), dim=0)
                        z21_g = torch.cat(FullGatherLayer.apply(encoder.module.simclr_proj(z21)), dim=0)
                        feat = torch.cat([z12_g, z21_g, h2_mean_g, h1_mean_g], dim=0)
                        loss_extra = info_nce_loss(feat)
                        recalls = compute_recall_at_k(feat[:feat.shape[0]//2], feat[feat.shape[0]//2:], ks=[1,5,10])
                    else:
                        loss_extra = (loss_fn(z12, h2_mean) + loss_fn(z21, h1_mean)) / 2
                        recalls = compute_recall_at_k(torch.cat([z12.detach(), z21.detach()], dim=0), torch.cat([h2_mean.detach(), h1_mean.detach()], dim=0), ks=[1,5,10], gather=True)
                    loss = args.intra_weight * loss_intra + args.extra_weight * loss_extra
                    
                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats, float(loss_intra), float(loss_extra), recalls[1], recalls[5], recalls[10])
            (loss, _new_lr, _new_wd, grad_stats, loss_intra, loss_extra, recall1, recall5, recall10), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            loss_intra_meter.update(loss_intra)
            loss_extra_meter.update(loss_extra)
            recall1_meter.update(recall1)
            recall5_meter.update(recall5)
            recall10_meter.update(recall10)
            
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))
                    logger.info(f'[{epoch+1}, {itr}] '
                                f'loss_intra: {loss_intra_meter.avg:.3f} '
                                f'loss_extra: {loss_extra_meter.avg:.3f} '
                                f'recall@1: {recall1_meter.avg:.3f} '
                                f'recall@5: {recall5_meter.avg:.3f} '
                                f'recall@10: {recall10_meter.avg:.3f} '
                            )
            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)


if __name__ == "__main__":
    main()
