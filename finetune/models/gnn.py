import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class RelationshipFusionWithoutAttention(nn.Module):
    def __init__(self, feature_dim, relationship_feature_dim, cfg='mask'):
        super().__init__()

        # Message MLP without attention mechanism
        if 'msgdual' in cfg:
            self.message_mlp = nn.Sequential(
                nn.Linear(2 * feature_dim + relationship_feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
            )
        else:
            self.message_mlp = nn.Sequential(
                nn.Linear(feature_dim + relationship_feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
            )

        # Optional gating mechanism for message update
        self.cfg = cfg
        if 'gate' in cfg:
            self.gate_proj = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid(),
            )


    def forward(self, node_features, relationship_features):
        """
        node_features: Tensor of shape (batch_size, N, node_feature_dim)
        relationship_features: Tensor of shape (batch_size, N, N, relationship_feature_dim)
        """
        batch_size, N, _ = node_features.size()

        # Prepare node features for source and target nodes
        x_i = node_features.unsqueeze(2).repeat(1, 1, N, 1)  # Shape: (batch_size, N, N, node_feature_dim)
        x_j = node_features.unsqueeze(1).repeat(1, N, 1, 1)  # Shape: (batch_size, N, N, node_feature_dim)

        # Flatten tensors to shape (batch_size, N*N, feature_dim)
        x_i_flat = x_i.reshape(batch_size, N * N, -1)  # Shape: (batch_size, N*N, node_feature_dim)
        x_j_flat = x_j.reshape(batch_size, N * N, -1)  # Shape: (batch_size, N*N, node_feature_dim)
        R_ij_flat = relationship_features.reshape(batch_size, N * N, -1)  # Shape: (batch_size, N*N, relationship_feature_dim)

        # Compute messages m_ij (no attention, just message passing)
        if 'msgdual' in self.cfg:
            msg_input = torch.cat([x_i_flat, x_j_flat, R_ij_flat], dim=2)
        else:
            msg_input = torch.cat([x_j_flat, R_ij_flat], dim=2)  # Shape: (batch_size, N*N, node_feature_dim + relationship_feature_dim)

        messages = self.message_mlp(msg_input)  # Shape: (batch_size, N*N, message_dim)

        # Aggregate messages (mean aggregation over neighbors)
        messages_reshaped = messages.view(batch_size, N, N, -1)  # Shape: (batch_size, N, N, message_dim)
        aggregated_messages = messages_reshaped.mean(dim=2)  # Shape: (batch_size, N, message_dim)

        updated_node_features = node_features + aggregated_messages

        return updated_node_features


class RelationshipFusionWithAttention(nn.Module):
    def __init__(self, feature_dim, relationship_feature_dim, cfg='mask'):
        super().__init__()
        # self.relationship_encoder = nn.Sequential(
        #     nn.Linear(relationship_feature_dim, feature_dim),
        #     nn.GELU(),
        #     nn.Linear(feature_dim, feature_dim)
        # )
        # Linear transformations for attention mechanism
        if 'attnmsg' in cfg:
            self.attention_mlp = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, 1)
            )
        elif 'msgdual' in cfg:
            self.attention_mlp = nn.Linear(feature_dim, 1)
        else:         
            self.attention_mlp = nn.Sequential(
                nn.Linear(2 * feature_dim + relationship_feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, 1)
            )
        
        # Message MLP
        if 'msgdual' in cfg:
            self.message_mlp = nn.Sequential(
                nn.Linear(2 * feature_dim + relationship_feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
            )
        else:
            self.message_mlp = nn.Sequential(
                nn.Linear(feature_dim + relationship_feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim),
            )
        self.cfg = cfg
        if 'gate' in cfg:
            self.gate_proj = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid(),
            )
        

    def forward(self, node_features, relationship_features):
        """
        node_features: Tensor of shape (batch_size, N, node_feature_dim)
        relationship_features: Tensor of shape (batch_size, N, N, relationship_feature_dim)
        """
        batch_size, N, _ = node_features.size()
        # Prepare node features for source and target nodes
        x_i = node_features.unsqueeze(2).repeat(1, 1, N, 1)  # Shape: (batch_size, N, N, node_feature_dim)
        x_j = node_features.unsqueeze(1).repeat(1, N, 1, 1)  # Shape: (batch_size, N, N, node_feature_dim)
        
        # Flatten tensors to shape (batch_size, N*N, feature_dim)
        x_i_flat = x_i.reshape(batch_size, N * N, -1)  # Shape: (batch_size, N*N, node_feature_dim)
        x_j_flat = x_j.reshape(batch_size, N * N, -1)  # Shape: (batch_size, N*N, node_feature_dim)
        R_ij_flat = relationship_features.reshape(batch_size, N * N, -1)  # Shape: (batch_size, N*N, relationship_feature_dim)
        

        
        # Compute messages m_ij
        if 'msgdual' in self.cfg:
            msg_input = torch.cat([x_i_flat, x_j_flat, R_ij_flat], dim=2)
        else:
            msg_input = torch.cat([x_j_flat, R_ij_flat], dim=2)  # Shape: (batch_size, N*N, node_feature_dim + relationship_feature_dim)
        messages = self.message_mlp(msg_input)  # Shape: (batch_size, N*N, message_dim)
        
        
        ## attn
        # Compute raw attention coefficients e_ij
        if 'msgdual' in self.cfg:
            att_input = messages
        elif 'attnmsg' in self.cfg:
            att_input = torch.cat([x_i_flat, messages], dim=2)
        else:
            att_input = torch.cat([x_i_flat, x_j_flat, R_ij_flat], dim=2)  # Shape: (batch_size, N*N, 2 * node_feature_dim + relationship_feature_dim)
        e_ij = self.attention_mlp(att_input).squeeze(-1)  # Shape: (batch_size, N*N)
        
        # Reshape e_ij to (batch_size, N, N)
        e_ij = e_ij.view(batch_size, N, N)
        
        if 'mask' in self.cfg:
            mask = torch.ones_like(e_ij)
            mask[:, range(N), range(N)] = 0  # Set diagonal to zero
            
            # Apply the mask before softmax
            e_ij = e_ij.masked_fill(mask == 0, float('-inf'))        
        
        # Apply softmax over the source nodes for each target node i in each graph
        alpha_ij = F.softmax(e_ij, dim=2)  # Shape: (batch_size, N, N)
        
        
        # Weight messages by attention scores
        alpha_ij_flat = alpha_ij.view(batch_size, N * N, 1)  # Shape: (batch_size, N*N, 1)
        weighted_messages = messages * alpha_ij_flat  # Shape: (batch_size, N*N, message_dim)
        
        # Reshape weighted_messages to (batch_size, N, N, message_dim)
        weighted_messages = weighted_messages.view(batch_size, N, N, -1)
        
        # Aggregate messages for each node i (sum over source nodes j)
        aggregated_messages = weighted_messages.sum(dim=2)  # Shape: (batch_size, N, message_dim)
        
        # Update node features
        if 'gate' in self.cfg:
            concat_feature = torch.cat([node_features, aggregated_messages], dim=-1)
            gate_values = self.gate_proj(concat_feature)
            updated_node_features = node_features * gate_values + aggregated_messages * (1-gate_values)
        elif 'drop' in self.cfg:
            updated_node_features = aggregated_messages
        else:
            updated_node_features = node_features + aggregated_messages
        
        
        return updated_node_features



class GeometricAttention(nn.Module):
    def __init__(self, feature_dim, rel_pos_dim, num_heads=4):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0, "Feature dimension must be divisible by the number of heads."
        
        
        self.query_linear = nn.Linear(feature_dim, feature_dim)
        self.key_linear = nn.Linear(feature_dim + rel_pos_dim, feature_dim)
        self.value_linear = nn.Linear(feature_dim + rel_pos_dim, feature_dim)
        
        self.output_linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, node_features, relative_positions):
        batch_size, N, _ = node_features.size()
        
        # Compute queries from node features
        Q = self.query_linear(node_features).view(batch_size, N, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # Shape: (batch_size, num_heads, N, head_dim)
        
        # Concatenate node features with relative position features for K and V
        K_input = torch.cat([node_features.unsqueeze(1).expand(-1, N, -1, -1), relative_positions], dim=-1)
        V_input = torch.cat([node_features.unsqueeze(1).expand(-1, N, -1, -1), relative_positions], dim=-1)
        
        K = self.key_linear(K_input).view(batch_size, N, N, self.num_heads, self.head_dim)
        V = self.value_linear(V_input).view(batch_size, N, N, self.num_heads, self.head_dim)
        
        # Transpose K and V for dot product compatibility
        K = K.permute(0, 3, 1, 2, 4)  # (batch_size, num_heads, N, N, head_dim)
        V = V.permute(0, 3, 1, 2, 4)  # (batch_size, num_heads, N, N, head_dim)
        
        # Dot-product attention with corrected einsum
        scores = torch.einsum('bhid,bhijd->bhij', Q, K) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values with corrected einsum
        weighted_values = torch.einsum('bhij,bhijd->bhid', attn_weights, V)
        
        # Reshape and output
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, N, -1)
        output = self.output_linear(weighted_values)
        
        output = node_features + output
        return output


class GeometricAttentionMLP(nn.Module):
    def __init__(self, feature_dim, rel_pos_dim, num_heads=4):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        assert feature_dim % num_heads == 0, "Feature dimension must be divisible by the number of heads."
        
        
        self.query_linear = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim)
            )
        self.key_linear = nn.Sequential(
                nn.Linear(feature_dim + rel_pos_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim)
            )
        self.value_linear = nn.Sequential(
                nn.Linear(feature_dim + rel_pos_dim, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, feature_dim)
            )
        # self.query_linear = nn.Linear(feature_dim, feature_dim)
        # self.key_linear = nn.Linear(feature_dim + rel_pos_dim, feature_dim)
        # self.value_linear = nn.Linear(feature_dim + rel_pos_dim, feature_dim)
        
        self.output_linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, node_features, relative_positions, return_attn=False):
        batch_size, N, _ = node_features.size()
        
        # Compute queries from node features
        Q = self.query_linear(node_features).view(batch_size, N, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # Shape: (batch_size, num_heads, N, head_dim)
        
        # Concatenate node features with relative position features for K and V
        K_input = torch.cat([node_features.unsqueeze(1).expand(-1, N, -1, -1), relative_positions], dim=-1)
        V_input = torch.cat([node_features.unsqueeze(1).expand(-1, N, -1, -1), relative_positions], dim=-1)
        
        K = self.key_linear(K_input).view(batch_size, N, N, self.num_heads, self.head_dim)
        V = self.value_linear(V_input).view(batch_size, N, N, self.num_heads, self.head_dim)
        
        # Transpose K and V for dot product compatibility
        K = K.permute(0, 3, 1, 2, 4)  # (batch_size, num_heads, N, N, head_dim)
        V = V.permute(0, 3, 1, 2, 4)  # (batch_size, num_heads, N, N, head_dim)
        
        # Dot-product attention with corrected einsum
        scores = torch.einsum('bhid,bhijd->bhij', Q, K) / (self.head_dim ** 0.5)
        

        attn_weights = F.softmax(scores, dim=-1)
        # Apply attention weights to values with corrected einsum
        weighted_values = torch.einsum('bhij,bhijd->bhid', attn_weights, V)
        
        # Reshape and output
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, N, -1)
        output = self.output_linear(weighted_values)
        
        output = node_features + output
        if return_attn:
            return output, attn_weights
        
        return output