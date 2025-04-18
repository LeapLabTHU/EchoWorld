from torch import nn



def build_proj(proj_type, in_dim, out_dim):
    if proj_type == 'mlp' or proj_type == 'mlp2':
        linear = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.GELU(),
                nn.Linear(in_dim // 2, out_dim)
            )
    elif proj_type == 'mlp3':
        linear = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.GELU(),
                nn.Linear(in_dim // 2, in_dim // 4),
                nn.GELU(),
                nn.Linear(in_dim // 4, out_dim)
            )
    elif proj_type == 'mlp4':
        linear = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.GELU(),
                nn.Linear(in_dim // 2, in_dim // 4),
                nn.GELU(),
                nn.Linear(in_dim // 4, in_dim // 8),
                nn.GELU(),
                nn.Linear(in_dim // 8, out_dim)
            ) 
    elif proj_type == 'fc':
        linear = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    return linear