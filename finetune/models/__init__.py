import torch
import logging
import os
from .proj import build_proj
from torch import nn
from einops import rearrange


def load_dinov2_model_official(args):
    from timm.models.vision_transformer import vit_small_patch14_dinov2, checkpoint_filter_fn
    encoder = vit_small_patch14_dinov2(num_classes=0, img_size=224)
    if args.pretrained == '':
        return encoder
    else:
        sd = torch.load(args.pretrained, map_location='cpu')
        sd = checkpoint_filter_fn(sd, encoder)
        msg = encoder.load_state_dict(sd, strict=False)
        logging.info(f'Load Dino Official from {args.pretrained} with msg {msg}')
        return encoder 

def load_deit(args):
    from timm.models.vision_transformer import vit_small_patch16_224
    encoder = vit_small_patch16_224(num_classes=0)
    sd = torch.load('checkpoint/deit_small_patch16_224-cd65a155.pth', map_location='cpu')['model']
    msg = encoder.load_state_dict(sd, strict=False)
    logging.info(f'Load DeiT from {args.pretrained} with msg {msg}')
    return encoder

def load_ijepa_model(args):
    import models.vision_transformer as vits
    encoder = vits.__dict__['vit_small'](img_size=[224],
            patch_size=16, drop_path_rate=args.drop_path)
    if args.pretrained == '':
        return encoder
    ckpt = torch.load(args.pretrained, map_location='cpu')
    if 'target_encoder' in ckpt:
        state_dict = ckpt["target_encoder"]
    elif 'model' in ckpt:
        state_dict = ckpt['model']
        state_dict = {k.replace('target_encoder.', ''): v for k, v in state_dict.items() if k.startswith('target_encoder.')}
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            # 去掉 'module.' 前缀
            new_key = k[7:]  # 跳过前7个字符 'module.'
        else:
            new_key = k
        new_state_dict[new_key] = v
    msg = encoder.load_state_dict(new_state_dict, strict=False)
    logging.info(f'Load Pretrained {os.path.basename(args.pretrained)} with missing {msg.missing_keys}')
    return encoder
    
    
def load_dinov2_model(args):
    import dinov2.models.vision_transformer as vits
    vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0e-05,
            ffn_layer="mlp", # if args.model == 'vit_base' else "swiglufused"
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            drop_path_rate=args.drop_path
        )
    model = vits.__dict__['vit_base'](**vit_kwargs)

    sd = torch.load(args.pretrained, map_location="cpu")
    if 'teacher' in sd:
        sd = sd['teacher']
        sd = {k.replace('backbone.', ''): v for k,v in sd.items() if 'backbone.' in k}
    msg = model.load_state_dict(sd, strict=False)
    logging.info(f'Missing: {str(msg.missing_keys)}')
    logging.info(f'Load Pretrained {os.path.basename(args.pretrained)}')
    return model


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, feature_model, num_classes=6, proj_type='mlp', pred_mode='euler'):
        super().__init__()
        self.feature_model = feature_model
        if hasattr(feature_model, 'embed_dim'):
            self.out_dim = feature_model.embed_dim
        else:
            self.out_dim = feature_model.num_features
        self.num_classes = num_classes
        self.linear = nn.ModuleList([build_proj(proj_type, self.out_dim, num_classes) for _ in range(10)])

    def forward(self, img):
        output = self.feature_model(img)
        if output.ndim == 3:
            output = output.mean(1) # average over seq
        outputs = [fc(output) for fc in self.linear]  # list of 10 * (b, 6)
        outputs = torch.stack(outputs)  # (10, b, 6)
        outputs = rearrange(outputs, 'f b n -> b (f n)')  # (b, 10*6)
        return outputs



class AttentionRPE(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rpe):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale + rpe
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultiFrameClassifier(nn.Module):
    def __init__(self, feature_model, num_classes=6, num_heads=4, proj_type='mlp'):
        super().__init__()
        self.feature_model = feature_model
        self.out_dim = feature_model.embed_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.linear = nn.ModuleList([build_proj(proj_type, self.out_dim, num_classes) for _ in range(10)])
        self.action_encoder = nn.Sequential(
            nn.Linear(num_classes, self.out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.out_dim, num_heads, bias=True)
        )
        nn.init.constant_(self.action_encoder[-1].weight, 0) # no rpb at first
        nn.init.constant_(self.action_encoder[-1].bias, 0)
        self.attn = AttentionRPE(dim=self.out_dim, num_heads=num_heads, qkv_bias=True)

    def forward(self, imgs, acts):
        '''
        imgs: [B, N, C, H, W], acts: [B, N, N, 6]
        '''
        B, N = imgs.shape[0], imgs.shape[1]
        rpe_vals = self.action_encoder(acts).permute(0, 3, 1, 2) # [B, H, N, N]
        img_feats = self.feature_model(imgs.view(-1, *imgs.shape[2:]))
        if img_feats.ndim == 3:
            img_feats = img_feats.mean(1) 
        # feature [B*N, D]
        img_feats = img_feats.view(B, N, -1) # [B, N, D]
        img_feats = img_feats + self.attn(img_feats, rpe_vals) #[B, N, D]
        img_feats = img_feats.view(B*N, -1) #[B*N, D]
        outputs = [fc(img_feats) for fc in self.linear]  # list of 10 * (b, 6)
        outputs = torch.stack(outputs)  # (10, b, 6)
        outputs = rearrange(outputs, 'f b n -> b (f n)')  # (b, 10*6)
        return outputs.view(B, N, -1)
        




from .gnn import RelationshipFusionWithAttention, GeometricAttention,  GeometricAttentionMLP, RelationshipFusionWithoutAttention
import torch.nn.functional as F
        
class MultiFrameClassifierGNN(nn.Module):
    def __init__(self, feature_model, num_classes=6, proj_type='mlp', gnn_cfg='mask'):
        super().__init__()
        self.feature_model = feature_model
        self.out_dim = feature_model.embed_dim
        self.num_classes = num_classes
        self.linear = nn.ModuleList([build_proj(proj_type, self.out_dim, num_classes) for _ in range(10)])

        self.action_encoder = nn.Sequential(
            nn.Linear(num_classes, self.out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(self.out_dim, self.out_dim, bias=True)
        )
        if 'geo' in gnn_cfg:
            if 'geomlp' in gnn_cfg:
                self.attn = GeometricAttentionMLP(feature_dim=self.out_dim, rel_pos_dim=self.out_dim)
            else:
                self.attn = GeometricAttention(feature_dim=self.out_dim, rel_pos_dim=self.out_dim)
        elif 'attn' in gnn_cfg:
            self.attn = RelationshipFusionWithAttention(feature_dim=self.out_dim, relationship_feature_dim=self.out_dim, cfg=gnn_cfg)
        else:
            self.attn = RelationshipFusionWithoutAttention(feature_dim=self.out_dim, relationship_feature_dim=self.out_dim, cfg=gnn_cfg)
        
        logging.info(f'GNN CFG: {gnn_cfg} {type(self.attn)}')

    def forward(self, imgs, acts):
        '''
        imgs: [B, N, C, H, W], acts: [B, N, N, 6]
        '''
        B, N = imgs.shape[0], imgs.shape[1]
        img_feats = self.feature_model(imgs.view(-1, *imgs.shape[2:]))
        if img_feats.ndim == 3:
            img_feats = img_feats.mean(1) 
        act_feats = self.action_encoder(acts)
        # feature [B*N, D]
        img_feats = img_feats.view(B, N, -1) # [B, N, D]
        
        img_feats = self.attn(img_feats, act_feats) #[B, N, D]
        
        img_feats = img_feats.view(B*N, -1) #[B*N, D]
        outputs = [fc(img_feats) for fc in self.linear]  # list of 10 * (b, 6)
        outputs = torch.stack(outputs)  # (10, b, 6)
        outputs = rearrange(outputs, 'f b n -> b (f n)')  # (b, 10*6)
        
        return outputs.view(B, N, -1)


    def forward_with_plane_cls(self, imgs, acts, plane_cls_feats):
        '''
        imgs: [B, N, C, H, W], acts: [B, N, N, 6]
        '''
        B, N = imgs.shape[0], imgs.shape[1]
        img_feats = self.feature_model(imgs.view(-1, *imgs.shape[2:]))
        if img_feats.ndim == 3:
            img_feats = img_feats.mean(1) 
        act_feats = self.action_encoder(acts)
        # feature [B*N, D]
        visual_feats = F.normalize(self.proj_visual(img_feats), dim=-1)
        plane_cls_feats = F.normalize(self.proj_plane_cls(plane_cls_feats), dim=-1)
        
        
        
        
        img_feats = img_feats.view(B, N, -1) # [B, N, D]
        img_feats = self.attn(img_feats, act_feats) #[B, N, D]
        
        img_feats = img_feats.view(B*N, -1) #[B*N, D]
        outputs = [fc(img_feats) for fc in self.linear]  # list of 10 * (b, 6)
        outputs = torch.stack(outputs)  # (10, b, 6)
        outputs = rearrange(outputs, 'f b n -> b (f n)')  # (b, 10*6)
        
        return outputs.view(B, N, -1), (visual_feats, plane_cls_feats)