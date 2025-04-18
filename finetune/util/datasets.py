import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import logging

def get_loader(args, dataset, is_train=True, persistent_workers=True, custom_bs=None, dist_eval=False,):
    if is_train or dist_eval:
        num_tasks = args.world_size
        global_rank = args.rank
        sampler = data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=is_train
        )
    else:
        sampler = data.SequentialSampler(dataset)
    
    # this_bs = args.batch_size if is_train else args.batch_size * 2
    this_bs = args.batch_size
    this_bs = custom_bs if custom_bs is not None else this_bs
    return data.DataLoader(
        dataset, sampler=sampler,
        batch_size=this_bs,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=is_train,
        persistent_workers=persistent_workers
    )
