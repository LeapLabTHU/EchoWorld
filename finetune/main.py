import argparse
import datetime
import json
import numpy as np
import os
import time
import math
import sys
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.utils.data
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from copy import deepcopy
# from torch.utils.tensorboard import SummaryWriter
import logging
from torchvision import transforms
#assert timm.__version__ == "0.3.2"  # version check
# import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.logger import create_logger, AverageMeter, ProgressMeter, MaxMeter

from util.misc import NativeScalerWithGradNormCount as NativeScaler

from util import build_optimizer, param_groups_lrd, add_weight_decay
from util.losses import MaskedSmoothL1Loss, MaskedSmoothL1LossEqual

from util.datasets import get_loader
from util.transforms import build_transform
from util.metrics import cal_metrics
from echo_datasets_new import MultiFrameDatasetPolicy
from einops import rearrange
from models.proj import build_proj
from util.transform_conversion import *
from opts import create_parser
from models import load_dinov2_model, load_ijepa_model, LinearClassifier, load_deit, MultiFrameClassifier, MultiFrameClassifierGNN
from autous import Transformation
from tqdm import tqdm

# from ensemble import dynamic_ensemble, median_ensemble
def action_ensemble(outputs, rel_pos):
    def concat_action(hexa1, hexa2):
        T = Transformation.hexa2trans(hexa1) @ Transformation.hexa2trans(hexa2)
        return Transformation.trans2hexa(T)
    '''Ensemble and output the prediction for the last position'''

    B, N, _ = outputs.shape
    res = torch.zeros((B, 10, 6), device=outputs.device)
    outputs = outputs.view(B, N, 10, 6)
    for b in range(B):
        for p in range(10):
            # from last frame to all other frames
            acts = [concat_action(rel_pos[b, n].cpu().numpy(), outputs[b, n, p].cpu().numpy()) for n in range(N)]
            acts = sum(acts) / len(acts)
            # acts = median_ensemble(acts)
            # acts = dynamic_ensemble(acts)
            res[b, p] = torch.tensor(acts, device=res.device)
    return res.flatten(1)

JHJ_TRANS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.193, 0.193, 0.193], [0.224, 0.224, 0.224])
])

@torch.no_grad()
def validate_ar(model, data_loader, args=None, debug=False):
    start_time = time.time()
    ae = torch.zeros(60, dtype=torch.float)
    ens_ae = torch.zeros(60, dtype=torch.float)
    ae_count = torch.zeros(60, dtype=torch.long)

    
    if debug:
        all_preds = []
        all_preds_ens = []
        all_targets = []
    for i, batch in enumerate(tqdm(data_loader)):
        images = batch[0]
        acts = batch[1].float() / args.label_scale
        if args.zero_act:
            acts = torch.zeros_like(acts)
        target = batch[2].float()
        masks = batch[3].float()
        images = images.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        # masks = masks.cuda(non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(images, acts) # [B, N, 60]
            if args.pred_every:
                action_preds = outputs[:, -1] # only keep the last one
                rel_pos = batch[4].float()
                action_preds_ens = action_ensemble(outputs, rel_pos)
            else:
                action_preds = outputs
                action_preds_ens = torch.zeros_like(action_preds)                    
            action_preds = action_preds.cpu()
            action_preds_ens=  action_preds_ens.cpu()
            # loss_real = torch.abs(action_preds - target)

        masked_targets = target * masks
        masked_pred = action_preds * masks
        masked_pred_ens = action_preds_ens * masks

        # if debug:
        #     all_preds.append(masked_pred)
        #     all_preds_ens.append(masked_pred_ens)
        #     all_targets.append(masked_targets)
        
        ae_count = ae_count + masks.sum(0)
        ae += torch.abs(masked_targets.float() - masked_pred).sum(0) # (10*6)
        ens_ae += torch.abs(masked_targets.float() - masked_pred_ens).sum(0) # (10*6)

    # if debug:
    #     all_preds = torch.cat(all_preds, dim=0).view(-1, 10, 6)
    #     all_preds_ens = torch.cat(all_preds_ens, dim=0).view(-1, 10, 6)
    #     all_targets = torch.cat(all_targets, dim=0).view(-1, 10, 6)
    #     all_ae = (all_preds - all_targets).abs().sum(-1)
    #     all_ae_ens = (all_targets - all_preds_ens).abs().sum(-1)
    #     import matplotlib.pyplot as plt
    #     for plane in range(10):
    #         plt.plot(all_ae[:, plane].numpy(), label='policy_ae')
    #         plt.plot(all_ae_ens[:, plane].numpy(), label='ensemble_ae')
    #         plt.legend()
    #         plt.savefig(f'{args.output_dir}/ae_p{plane}.png')
    #         plt.close()
    mae = (ae / ae_count).view(10, 6).numpy()
    mae_ens = (ens_ae / ae_count).view(10, 6).numpy()
    # agent_mae = (agent_ae / ae_count).view(10, 6).numpy()
    report_mae = mae_ens if 'gnn' in args.model else mae
    logging.info('===Policy===')
    for c in range(0, 10):
        val = '  '.join([f'{v:.4f}' for v in mae[c]])
        logging.info(f'Plane#{c+1}: {val}')
    logging.info('===Ensemble===')
    for c in range(0, 10):
        val = '  '.join([f'{v:.4f}' for v in mae_ens[c]])
        logging.info(f'Plane#{c+1}: {val}')
    logging.info(f'element count: {ae_count.sum().item()}')
    logging.info(f'Policy-MAE: {np.nanmean(mae):.4f}, trans_rot={mae.reshape(10, 2, 3).mean((0,2))}')
    logging.info(f'Ensemble-MAE: {np.nanmean(mae_ens):.4f}, trans_rot={mae_ens.reshape(10, 2, 3).mean((0,2))}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info('Elapsed: {}'.format(total_time_str))
    return np.nanmean(report_mae)



def main(args):
    misc.init_dist_pytorch(args)
    # args.rank = 0
    create_logger(args)
    logging.info(f'job dir: {args.output_dir}')
    device = torch.device('cuda')

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    pred_every = 'aware' not in args.model
    args.pred_every = pred_every
    if 'gnn' in args.model:
        act_mode = 'full'
    elif 'aware' in args.model:
        act_mode = 'seq_inv'
    else:
        act_mode = 'seq'
    args.act_mode = act_mode
    logging.info(f'pred_every: {pred_every}, act_mode: {act_mode}')
    

    if args.cross_val_fold != -1:
        dataset_train = MultiFrameDatasetPolicy(num_frames=args.num_frames, split='train', transform=build_transform(args, train=True), num_repeat=args.dataset_cat, seq_prob=0.5, pred_every=pred_every, act_mode=act_mode, exp_alpha=args.exp_alpha, cross_val_fold=args.cross_val_fold, cross_val_train=True, nfolds=args.nfolds, ar_train=args.ar_train)
    else:
        dataset_train = MultiFrameDatasetPolicy(num_frames=args.num_frames, split='train', transform=build_transform(args, train=True), num_repeat=args.dataset_cat, seq_prob=0.5, pred_every=pred_every, act_mode=act_mode, exp_alpha=args.exp_alpha, ar_train=args.ar_train)
         
    # dataset_val = MultiFrameDatasetPolicy(num_frames=args.num_frames, num_repeat=10, split='val', transform=build_transform(args, train=False), stride=args.stride, seq_prob=1.0, pred_every=pred_every, act_mode=act_mode, exp_alpha=args.exp_alpha)

    if len(args.set_case_id) > 0:
        if os.path.exists(args.set_case_id[0]):
            with open(args.set_case_id[0], 'r') as f:
                lines = f.readlines()
                args.set_case_id = [l.strip() for l in lines if l.strip()]
        elif 'full' in args.set_case_id:
            args.set_case_id = []
    else:
        # default to 3 case eval
        args.set_case_id = ['2024-03-07_14-09-57', '2024-03-06_09-36-18', '2024-03-09_09-45-37']
    
    val_transform = JHJ_TRANS if args.model == 'aware_tuned' else build_transform(args, train=False)
    dataset_val_ar = MultiFrameDatasetPolicy(num_frames=args.num_frames, num_repeat=1, split='val', transform=val_transform, stride=args.stride, seq_prob=1.0, pred_every=pred_every, act_mode=act_mode, exp_alpha=args.exp_alpha, ar_eval=True, set_case_id=args.set_case_id)
    args.num_classes = 6
    data_loader_train = get_loader(args, dataset_train, is_train=True)
    data_loader_val = get_loader(args, dataset_val_ar, is_train=False)

    
    if args.model == 'single':
        from models.seq_models import SeqWrapper
        model = load_ijepa_model(args)
        model = LinearClassifier(model, proj_type=args.proj_type)
        sd = torch.load(args.pretrained, map_location='cpu')['model']
        model.load_state_dict(sd)
        model = SeqWrapper(model)
    elif args.model == 'aware_tuned':
        from models.jhj_models import ViT_Cardiac_Seq_Model
        model = ViT_Cardiac_Seq_Model(
            model_name='vit_small',
            timestep=args.num_frames,
            modelpath=None,
            encoderpath=None,
            pred_depth=4,
            pred_emb_dim=192,
            pred_num_heads=4,
            pred_mlp_ratio=2
        )
        sd = torch.load(args.pretrained, map_location='cpu')
        sd = {k.replace('encoder.', 'feature_model.'): v for k, v in sd.items()}
        model.load_state_dict(sd)
        args.label_scale = 1
    elif 'aware' in args.model:
        from models.jhj_models import ViT_Cardiac_Seq_Model
        model = ViT_Cardiac_Seq_Model(
            model_name='vit_small',
            timestep=args.num_frames,
            modelpath=args.pretrained,
            encoderpath=None,
            pred_depth=4,
            pred_emb_dim=192,
            pred_num_heads=4,
            pred_mlp_ratio=2
        )
        for p in model.feature_model.parameters():
            p.requires_grad = False
        args.label_scale = 1
    else:
        args.label_scale = 200
        # use my own pretrained model
        if 'deit' in args.model:
            base_model = load_deit(args)
        else:
            base_model = load_ijepa_model(args)
        
        if 'gnn' in args.model:
            model = MultiFrameClassifierGNN(base_model, proj_type=args.proj_type, gnn_cfg=args.gnn_cfg)
        elif 'guide' in args.model:
            from models.seq_models import US_GuideNet
            model = US_GuideNet(base_model, proj_type=args.proj_type)
        elif 'decision_v2' in args.model:
            from models.seq_models import DecisionTransformerV2
            model = DecisionTransformerV2(base_model, proj_type=args.proj_type, max_frames=args.num_frames)
        elif 'decision' in args.model:
            from models.seq_models import DecisionTransformer
            model = DecisionTransformer(base_model, proj_type=args.proj_type, max_frames=args.num_frames)
        else:
            raise NotImplementedError
    
    
        if args.pretrained and 'gnn' in args.model:
            # only our model can use the pretrained policy
            ckpt = torch.load(args.pretrained, map_location='cpu')
            if 'predictor' in ckpt:
                policy_net_sd = {k.replace('module.policy_net.', ''): v for k, v in ckpt['predictor'].items() if k.startswith('module.policy_net.')}
                if len(policy_net_sd) > 0:
                    model.action_encoder.load_state_dict(policy_net_sd)
                    logging.info('Load Policy Net!')

    
    model.to(device)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    logging.info("actual lr: %.2e" % args.lr)

    logging.info("accumulate grad iterations: %d" % args.accum_iter)
    logging.info("effective batch size: %d" % eff_batch_size)

    if args.equal_loss:
        criterion = MaskedSmoothL1LossEqual()
    else:
        criterion = MaskedSmoothL1Loss()
    logging.info(f'criterion: {criterion}')

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    model_without_ddp = model.module


    if args.eval:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=None, loss_scaler=None, new_start=False)
        validate_ar(model, data_loader_val, args=args, debug=True)
        return
    
    if args.backbone_lr_scale < 1.0:
        backbone_param_groups = param_groups_lrd(model_without_ddp.feature_model, args.weight_decay, no_weight_decay_list=['pos_embed', 'cls_token', 'head.cls_query', 'prompt'], layer_decay=args.layer_decay)
        logging.info(f'Backbone lr scale = {args.backbone_lr_scale}')
        for group in backbone_param_groups:
            group['lr_scale'] = group['lr_scale'] * args.backbone_lr_scale
        head_param_groups = add_weight_decay(model_without_ddp.linear, weight_decay=args.weight_decay)
        param_groups = backbone_param_groups + head_param_groups
    else:
        param_groups = param_groups_lrd(model_without_ddp, args.weight_decay, no_weight_decay_list=['pos_embed', 'cls_token', 'head.cls_query', 'prompt'], layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), eps=1e-6)
    
    if args.amp:
        loss_scaler = NativeScaler()
    else:
        raise NotImplementedError
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, new_start=False)
    

    start_time = time.time()
    
    # best_state = None
    tolerence = 0

    val_meter = MaxMeter('MAE', mode='min')

    for epoch in range(args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        # data_time = AverageMeter('Data Time', ":6.3f")
        batch_time = AverageMeter('Time', ':6.3f')
        loss_meter = AverageMeter(f'Loss', ':.4f')
        meters = [batch_time, loss_meter]
        progress = ProgressMeter(
            len(data_loader_train),
            meters,
            prefix=f"[Epoch {epoch}] ")
        end = time.time()
        model.train(True)
        for data_iter_step, (samples, acts, targets, masks) in enumerate(data_loader_train):
            # data_time.update(time.time() - end)
            lr = adjust_learning_rate(optimizer, epoch + data_iter_step/len(data_loader_train), args)
            samples = samples.cuda(non_blocking=True)
            targets = targets.float().cuda(non_blocking=True)
            masks = masks.float().cuda(non_blocking=True)
            acts = acts.float().cuda(non_blocking=True) / args.label_scale # preproc
            if args.zero_act:
                acts = torch.zeros_like(acts)
            
            _b = samples.shape[0]
            with torch.cuda.amp.autocast():
                outputs = model(samples, acts)
                if args.pred_every and args.pred_last:
                    outputs, targets, masks = outputs[:, -1], targets[:, -1], masks[:, -1]
                loss = criterion(outputs, targets, masks)
            loss_value = loss.item()
            loss = loss / args.accum_iter
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            update_grad = (data_iter_step + 1) % args.accum_iter == 0
            if args.amp:
                loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                            parameters=model.parameters(), create_graph=False,
                            update_grad=update_grad)
            else:
                loss.backward()
                if update_grad:
                    if args.clip_grad is not None:
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            if update_grad:
                optimizer.zero_grad(set_to_none=True)
            # torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            loss_meter.update(loss.item(), args.batch_size)
        
            if data_iter_step % args.print_freq == 0:
                logging.info(progress.display(data_iter_step) + f'\tLR = {lr:.4e}')
            end = time.time()
        if args.rank == 0 and ((epoch+1) % args.eval_freq == 0 or epoch == args.epochs - 1):
            # val_stat = validate(model, data_loader_val, criterion, args=args)
            val_stat = validate_ar(model, data_loader_val, args=args)
            is_best = val_meter.update(val_stat, epoch)
            # val_aupr_meter.update(val_stat['maupr'], epoch)
            logging.info(val_meter.display())
            # logging.info(val_aupr_meter.display())
            if is_best:
                # best_state = deepcopy(model.state_dict())
                tolerence = 0
            else:
                tolerence += 1
            if args.rank == 0:
                to_save = {
                            'model': model_without_ddp.state_dict(),
                            'epoch': epoch,
                            'args': args
                            }
                if args.save_mode == 'best' and args.rank == 0 and is_best:
                    torch.save(to_save, os.path.join(args.output_dir, f'model_best.pth.tar'))
                if args.save_mode == 'every' and args.rank == 0:
                    torch.save(to_save, os.path.join(args.output_dir, f'epoch_{epoch}.pth.tar'))
        if args.rank == 0:
            misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch, is_best=False)
        # if args.early_stop is not None and tolerence >= args.early_stop:
        #     logging.info('Early Stop!')
        #     break
        end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info('Elapsed: {}'.format(total_time_str))
    if args.rank == 0:
        os.remove(os.path.join(args.output_dir, 'checkpoint.pth'))
    # logging.info('=== Testing ===')
    # model.load_state_dict(best_state)
    # test_stat = validate(model, data_loader_test, criterion, args=args, test=True)
    logging.info('=== Complete ===')


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if args.lr_sched == 'cos':
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs 
        else:
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    elif args.lr_sched == 'const':
        if epoch < args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs 
        else:
            lr = args.lr
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

if __name__ == '__main__':
    args = create_parser().parse_args()
    args.amp = True
    args.exp_name = os.path.join(args.exp_name, f'r{args.dataset_cat}x{args.epochs}_bs{args.batch_size}_dp{args.drop_path}_ld{args.layer_decay}_lr{args.lr}')
    args.output_dir = os.path.join(args.output_dir, args.exp_name, f'seed{args.seed}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
