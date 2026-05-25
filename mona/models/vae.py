import torch
import torch.nn as nn

class CausalConv3d(nn.Module):
    """
    A 3D Convolution layer that uses asymmetric padding along the temporal 
    dimension to ensure causal dependency (past frames can't see future frames).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # kernel_size is expected to be a tuple: (temporal, height, width)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
            
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Spatial padding is symmetric, temporal padding is manual and causal
        self.spatial_pad = (kernel_size[1] // 2, kernel_size[2] // 2)
        
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=(0, self.spatial_pad[0], self.spatial_pad[1])
        )

    def forward(self, x):
        # x shape: (batch, channels, time, height, width)
        # Apply manual causal padding to the temporal dimension (dim 2)
        # Pad only the front (left side) of the temporal sequence
        t_pad = self.kernel_size[0] - 1
        x = nn.functional.pad(x, (0, 0, 0, 0, t_pad, 0))
        
        return self.conv(x)

class SpatioTemporalVAEEncoder(nn.Module):
    """
    Compresses raw pixel videos down into compact latent representations.
    Reduces spatial resolution and temporal framerate.
    """
    def __init__(self, in_channels=3, latent_channels=4, hidden_dim=128):
        super().__init__()
        
        self.init_conv = CausalConv3d(in_channels, hidden_dim, kernel_size=3)
        
        # Downsampling block (Compresses space by 2x, time by 2x)
        self.downsample = nn.Sequential(
            CausalConv3d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=(2, 2, 2)),
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU()
        )
        
        # Project to Latent Space Mean and Log Variance
        self.to_moments = CausalConv3d(hidden_dim * 2, latent_channels * 2, kernel_size=1)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.downsample(x)
        moments = self.to_moments(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        return latent

class SpatioTemporalVAEDecoder(nn.Module):
    """
    Takes the denoised latent representations from the DiT and 
    upsamples them back into raw pixel video frames.
    """
    def __init__(self, latent_channels=4, out_channels=3, hidden_dim=256):
        super().__init__()
        
        self.init_latent = CausalConv3d(latent_channels, hidden_dim, kernel_size=1)
        
        # Upsampling blocks using 3D Transposed Convolutions
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, hidden_dim // 2, kernel_size=(3, 3, 3), 
                               stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1)),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU()
        )
        
        self.final_conv = CausalConv3d(hidden_dim // 2, out_channels, kernel_size=3)

    def forward(self, latent):
        x = self.init_latent(latent)
        x = self.upsample(x)
        x = self.final_conv(x)
        return torch.tanh(x) # Pixels scaled between [-1, 1]
