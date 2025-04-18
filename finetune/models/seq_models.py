
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from .proj import build_proj
from .vision_transformer import MLP, trunc_normal_

class US_GuideNet(nn.Module):
    def __init__(self, feature_model, model_dim=384, proj_type='mlp'):
        super().__init__()
        self.feature_model = feature_model
        self.model_dim = model_dim
        # self.fc1 = nn.Linear(feature_model.embed_dim, model_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(6, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
        self.gru = nn.GRU(input_size=model_dim * 2, hidden_size=model_dim, batch_first=True)
        self.fc_out  = nn.ModuleList([build_proj(proj_type, model_dim, 6) for _ in range(10)])
    
    def forward(self, imgs, acts):
        '''imgs: NxTxCxHxW, acts: Nx(T-1)x6'''
        N, T = imgs.shape[0], imgs.shape[1]
        img_feats = self.feature_model(imgs.view(-1, *imgs.shape[2:]))
        if img_feats.ndim == 3:
            img_feats = img_feats.mean(1)
        img_feats = img_feats.view(N, T, -1) #[N, T, D]
        # pad last zero action
        acts = torch.cat([torch.zeros((N, 1, 6), device=acts.device), acts], dim=1) #[N,T,6]
        act_feats = self.action_encoder(acts) #[N, T, D]
        concat_feats = torch.cat([img_feats, act_feats], dim=-1)
        gru_out, _ = self.gru(concat_feats)
        gru_out = gru_out.reshape(N*T, -1)
        outputs = [fc(gru_out) for fc in self.fc_out]  # list of 10 * (b, 6)
        outputs = torch.stack(outputs)  # (10, b, 6)
        outputs = rearrange(outputs, 'f b n -> b (f n)')  # (b, 10*6)
        
        return outputs.reshape(N, T, -1)




class CausalAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        use_sdpa=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        with torch.backends.cuda.sdp_kernel():
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.proj_drop_prob, is_causal=True)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CausalBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CausalAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class DecisionTransformer(nn.Module):
    def __init__(self, feature_model, max_frames=5, num_blocks=2, model_dim=384, proj_type='mlp', drop=0):
        super().__init__()
        self.feature_model = feature_model
        self.model_dim = model_dim
        self.fc1 = nn.Linear(feature_model.embed_dim, model_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(6, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
        self.fc_out  = nn.ModuleList([build_proj(proj_type, model_dim, 6) for _ in range(10)])
        self.blocks = nn.Sequential(*[CausalBlock(model_dim, num_heads=6, qkv_bias=True, drop=drop, attn_drop=drop) for _ in range(num_blocks)])
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * max_frames - 1, model_dim))
        trunc_normal_(self.pos_emb)
    
    def forward(self, imgs, acts):
        '''imgs: NxTxCxHxW, acts: Nx(T-1)x6'''
        N, T = imgs.shape[0], imgs.shape[1]
        img_feats = self.feature_model(imgs.view(-1, *imgs.shape[2:]))
        if img_feats.ndim == 3:
            img_feats = img_feats.mean(1)
        img_feats = self.fc1(img_feats).view(N, T, -1) #[N, T, D]
        # pad last zero action
        act_feats = self.action_encoder(acts) #[N, T-1, D]

        concat_feats = torch.zeros((N, 2*T-1, self.model_dim)).to(act_feats.device)
        concat_feats[:, ::2] = img_feats
        concat_feats[:, 1::2] = act_feats

        concat_feats = concat_feats + self.pos_emb.repeat(N, 1, 1)[:, :2*T-1]
        out = self.blocks(concat_feats)
        out_frame = out[:, ::2] #[N, T, D]
        out_frame = out_frame.reshape(N*T, -1)
        outputs = [fc(out_frame) for fc in self.fc_out]  # list of 10 * (b, 6)
        outputs = torch.stack(outputs)  # (10, b, 6)
        outputs = rearrange(outputs, 'f b n -> b (f n)')  # (b, 10*6)
        return outputs.reshape(N, T, -1)


class DecisionTransformerV2(nn.Module):
    def __init__(self, feature_model, max_frames=5, num_blocks=2, model_dim=384, proj_type='mlp', drop=0):
        super().__init__()
        self.feature_model = feature_model
        self.model_dim = model_dim
        self.fc1 = nn.Linear(feature_model.embed_dim, model_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(6, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )
        self.fc_out  = nn.ModuleList([build_proj(proj_type, model_dim, 6) for _ in range(10)])
        self.blocks = nn.Sequential(*[CausalBlock(model_dim, num_heads=6, qkv_bias=True, drop=drop, attn_drop=drop) for _ in range(num_blocks)])
        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames, model_dim))
        self.fuse_img_act = nn.Linear(model_dim * 2, model_dim)
        trunc_normal_(self.pos_emb)
    
    def forward(self, imgs, acts):
        '''imgs: NxTxCxHxW, acts: Nx(T-1)x6'''
        N, T = imgs.shape[0], imgs.shape[1]
        img_feats = self.feature_model(imgs.view(-1, *imgs.shape[2:]))
        if img_feats.ndim == 3:
            img_feats = img_feats.mean(1)
        img_feats = self.fc1(img_feats).view(N, T, -1) #[N, T, D]
        acts = torch.cat([torch.zeros((N, 1, 6), device=acts.device), acts], dim=1) #[N,T,6]
        act_feats = self.action_encoder(acts) #[N, T, D]
        concat_feats = torch.cat([img_feats, act_feats], dim=-1)
        concat_feats = self.fuse_img_act(concat_feats)
        concat_feats = concat_feats + self.pos_emb.repeat(N, 1, 1)[:, :T]
        out = self.blocks(concat_feats)
        out_frame = out.reshape(N*T, -1)
        outputs = [fc(out_frame) for fc in self.fc_out]  # list of 10 * (b, 6)
        outputs = torch.stack(outputs)  # (10, b, 6)
        outputs = rearrange(outputs, 'f b n -> b (f n)')  # (b, 10*6)
        return outputs.reshape(N, T, -1)



class SeqWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, imgs, acts):
        return self.base_model(imgs.view(-1, *imgs.shape[2:])).reshape(imgs.shape[0], imgs.shape[1], -1)