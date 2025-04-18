import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler
import torch.nn.functional as F

class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, z_channels=768, depth=3, width=768, grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.num_planes = 10
        self.action_dim = 6
        self.model = SimpleMLPAdaLN(
            in_channels=self.action_dim,
            model_channels=width,
            out_channels=self.action_dim,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

        # self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        # self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='squaredcos_cap_v2',
            prediction_type='sample',
            clip_sample=False,
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule='squaredcos_cap_v2',
            prediction_type='sample',
        )
        self.num_train_timesteps = 1000
        self.prediction_type = 'sample'
        


    # def forward(self, target, z, mask=None):
    #     t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
    #     model_kwargs = dict(c=z)
    #     loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
    #     loss = loss_dict["loss"]
    #     if mask is not None:
    #         loss = (loss * mask).sum() / mask.sum()
    #     return loss.mean()

    # def sample(self, z, temperature=1.0, cfg=1.0):
    #     # diffusion loss sampling
    #     if not cfg == 1.0:
    #         noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
    #         noise = torch.cat([noise, noise], dim=0)
    #         model_kwargs = dict(c=z, cfg_scale=cfg)
    #         sample_fn = self.net.forward_with_cfg
    #     else:
    #         noise = torch.randn(z.shape[0], self.in_channels).cuda()
    #         model_kwargs = dict(c=z)
    #         sample_fn = self.net.forward

    #     sampled_token_latent = self.gen_diffusion.p_sample_loop(
    #         sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
    #         temperature=temperature
    #     )

    #     return sampled_token_latent



    def conditional_sample(self, hiddens, sample_steps=5):
        batch_size = hiddens.shape[0]
        device = hiddens.device
        dtype = hiddens.dtype
        noisy_action = torch.randn(
            size=(batch_size * self.num_planes, self.action_dim), 
            dtype=dtype, device=device)
        plane_indices = torch.arange(self.num_planes, device=device).long().repeat(batch_size)
    
        # Set step values
        self.noise_scheduler_sample.set_timesteps(sample_steps)
        
        for t in self.noise_scheduler_sample.timesteps:
            timesteps = torch.ones((batch_size * self.num_planes,), device=device).long() * t
            hiddens_repeated = hiddens.unsqueeze(1).repeat(1, self.num_planes, 1).reshape(-1, hiddens.shape[-1])
            # Predict the model output
            model_output = self.model(noisy_action, timesteps, plane_indices, hiddens_repeated)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(device)

        return noisy_action.view(batch_size, -1) 
    
    # ========= Train  ============
    def compute_loss(self, hiddens, labels, masks) -> torch.Tensor:
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        batch_size = hiddens.shape[0]
        device = hiddens.device  
        
        labels = labels.reshape(-1, self.action_dim) # B*F, 6
        
        # Sample noise that we'll add to the actions
        noise = torch.randn(
            labels.shape, dtype=labels.dtype, device=device
        )
        # Sample random diffusion timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (batch_size * self.num_planes,), device=device
        ).long()
        plane_indices = torch.arange(self.num_planes, device=device).long().repeat(batch_size)
        
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(
            labels, noise, timesteps)
        
        
        hiddens_repeated = hiddens.unsqueeze(1).repeat(1, self.num_planes, 1).reshape(-1, hiddens.shape[-1])
        # Predict the denoised result
        pred = self.model(noisy_action, timesteps, plane_indices, hiddens_repeated)

        pred_type = self.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = labels
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = (loss.reshape(batch_size, -1) * masks).sum() / masks.sum() # filter out invalid
        return loss


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.plane_embed = nn.Embedding(10, model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, plane_idx, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        p = self.plane_embed(plane_idx)
        y = t + c + p

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class EchoMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        action_dim=6,
        embed_dim=384,
        model_channels=384,
        num_res_blocks=2,
        grad_checkpointing=False
    ):
        super().__init__()

        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.embed_dim = embed_dim


        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Initialize timestep embedding MLP
        # nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, y):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param y: an [N x A] Tensor of actions.
        :return: an [N x C] Tensor of outputs.
        """
        y = self.action_proj(y)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)
