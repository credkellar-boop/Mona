import torch

# Updated import path to point to the models directory
# Note: If your classes are inside specific files like `vae.py` and `dit.py` inside `models/`, 
# this would be: `from mona.models.vae import ...` and `from mona.models.dit import ...`
from mona.models import SpatioTemporalVAEEncoder, SpatioTemporalVAEDecoder, SpatioTemporalDiT

def test_vae_and_dit_shape_matching():
    # Safely falls back to CPU on GitHub Actions runner
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_video = torch.randn(1, 3, 24, 256, 256).to(device)

    # 1. Initialize the models FIRST
    encoder = SpatioTemporalVAEEncoder(in_channels=3, latent_channels=4, hidden_dim=32).to(device)
    decoder = SpatioTemporalVAEDecoder(latent_channels=4, out_channels=3, hidden_dim=64).to(device)
    dit = SpatioTemporalDiT(in_channels=4, hidden_size=128, depth=2, num_heads=2).to(device)

    # 2. Now you can use the encoder
    latents = encoder(dummy_video)
    b, c, t, h, w = latents.shape
    
    # 3. Complete your remaining patchification and testing assertions below
    assert latents is not None
