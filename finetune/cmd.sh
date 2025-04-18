torchrun --nproc_per_node 2 --nnodes 1  --rdzv_backend c10d --rdzv_endpoint localhost:0 \
main.py  --dataset_cat 400 --epochs 10 --warmup_epochs 1  --norm_type default   \
    --model gnn --pretrained <path-to-pretrain-ckpt> \
    --print_freq 20 --batch_size 32 --accum_iter 1 --num_workers 8 --eval_freq 1 \
    --input_size 224 --color_jitter 0.2 \
    --early_stop 15 --layer_decay 0.65 --drop_path 0.2 \
    --lr 5e-4 --min_lr 1e-6 --weight_decay 0.05 --clip_grad 1 --save_mode best \
    --stride 10 --num_frames 8 --exp_alpha 0.4 --ar_train --gnn_cfg geomlp \
    --output_dir output_dir  --seed 0 --exp_name exp_name 

