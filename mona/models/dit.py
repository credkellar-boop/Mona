import torch
import torch.nn as nn
import math

class ModulatedLayerNorm(nn.Module):
    """
    Adaptive LayerNorm (adaLN) that conditions the transformer 
    blocks on the timestep and textual prompt.
    """
    def __init__(self, hidden_size, condition_size):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_size, hidden_size * 6)
        )

    def forward(self, x, condition):
        # Generate scale and shift parameters from the conditioning
        shift, scale, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.mlp(condition).chunk(6, dim=-1)
        
        # Apply modulation
        x = self.ln(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x, gate_msa.unsqueeze(1), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1), gate_mlp.unsqueeze(1)

class SpatioTemporalAttention(nn.Module):
    """
    Processes both spatial patches and temporal frames simultaneously.
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.spatial_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward(self, x, frames=10):
        # 1. Spatial Attention (within the same frame)
        x_spatial, _ = self.spatial_attn(x, x, x)
        
        # 2. Temporal Attention (across frames)
        # Reshape to treat spatial patches as batch, and frames as sequence
        batch_size, seq_len, channels = x.shape
        patches_per_frame = seq_len // frames
        
        x_temporal = x.view(batch_size * patches_per_frame, frames, channels)
        x_temporal, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        
        # Merge back
        x_temporal = x_temporal.view(batch_size, seq_len, channels)
        
        return x_spatial + x_temporal

class DiTBlock(nn.Module):
    def __init__(self, hidden_size=1152, num_heads=16, condition_size=1152):
        super().__init__()
        self.norm1 = ModulatedLayerNorm(hidden_size, condition_size)
        self.attn = SpatioTemporalAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x, condition):
        x_norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(x, condition)
        
        # Attention with gating
        x = x + gate_msa * self.attn(x_norm)
        
        # MLP with modulation and gating
        x_mlp_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.mlp(x_mlp_norm)
        
        return x

class SpatioTemporalDiT(nn.Module):
    """
    The main Mona DiT model.
    """
    def __init__(self, in_channels=4, hidden_size=1152, depth=28, num_heads=16):
        super().__init__()
        
        # Patch embedding (converts latent image into transformer tokens)
        self.patch_embed = nn.Linear(in_channels * 4 * 4, hidden_size) 
        
        # The main transformer trunk
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, condition_size=hidden_size) 
            for _ in range(depth)
        ])
        
        # Final decoding layer
        self.final_layer = nn.Linear(hidden_size, in_channels * 4 * 4)

    def forward(self, x, condition):
        """
        x: Latent tensor from the VAE
        condition: Timestep + text embeddings from Gemini
        """
        x = self.patch_embed(x)
        
        for block in self.blocks:
            x = block(x, condition)
            
        x = self.final_layer(x)
        return x
