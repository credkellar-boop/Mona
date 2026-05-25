# train.py
import torch
from torch.utils.data import DataLoader
from data.dataset import MonaVideoDataset
from mona.models.dit import SpatioTemporalDiT
from mona.models.vae import SpatioTemporalVAEEncoder

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔥 Starting Mona training core on device: {device}")

    # 1. Initialize models
    vae_encoder = SpatioTemporalVAEEncoder(in_channels=3, latent_channels=4).to(device).eval() # Keep VAE frozen
    dit_model = SpatioTemporalDiT(in_channels=4, hidden_size=512, depth=12, num_heads=8).to(device).train()
    
    # 2. Setup Data Pipeline
    dataset = MonaVideoDataset(data_root="data/train_frames", annotation_file="data/annotations.txt")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    optimizer = torch.optim.AdamW(dit_model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # 3. Micro Training Loop Demo
    for epoch in range(1):
        for step, (videos, captions) in enumerate(dataloader):
            videos = videos.to(device) # (B, C, T, H, W)
            
            # Context-free guidance setup / Condition embedding placeholder
            # In production, Gemini text descriptions are passed through text-encoders like T5 or CLIP here
            dummy_condition = torch.randn(videos.shape[0], 512).to(device) 
            
            with torch.no_grad():
                # Compress video tokens into latent blocks across time/space via the Causal VAE
                latents = vae_encoder(videos) # Yields (B, Latent_C, T_compressed, H_compressed, W_compressed)
                
            # Flatten spatial/temporal dimensions into patches for the DiT sequence
            b, c, t, h, w = latents.shape
            latents_flattened = latents.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
            
            # Setup simple forward diffusion noise
            noise = torch.randn_like(latents_flattened)
            timesteps = torch.randint(0, 1000, (b,), device=device)
            
            # Forward Pass through Diffusion Transformer
            optimizer.zero_grad()
            model_output = dit_model(latents_flattened, dummy_condition)
            
            loss = criterion(model_output, noise) # Over-simplified score matching target
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
            break # Break immediately for testing framework layout

if __name__ == "__main__":
    train()
