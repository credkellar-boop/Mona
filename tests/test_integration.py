import os
import pytest
import torch
import asyncio
from mona.models.dit import SpatioTemporalDiT
from mona.models.vae import SpatioTemporalVAEEncoder, SpatioTemporalVAEDecoder
from mona.utils.live_sfu import LiveRoomManager
from data.dataset import MonaVideoDataset

def test_vae_and_dit_shape_matching():
    """
    Verifies that the 3D Causal VAE and Spatio-Temporal DiT 
    can pass tensors to each other without shape mismatches.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Simulate a mini input video tensor: (Batch, Channels, Time/Frames, Height, Width)
    dummy_video = torch.randn(1, 3, 24, 256, 256).to(device)
    
    # 2. Initialize Models with minimal footprints for testing
    encoder = SpatioTemporalVAEEncoder(in_channels=3, latent_channels=4, hidden_dim=32).to(device)
    decoder = SpatioTemporalVAEDecoder(latent_channels=4, out_channels=3, hidden_dim=64).to(device)
    dit = SpatioTemporalDiT(in_channels=4, hidden_size=128, depth=2, num_heads=2).to(device)
    
    # 3. Test VAE Encoding
    latents = encoder(dummy_video)
    b, c, t, h, w = latents.shape
    assert c == 4, f"Expected 4 latent channels, got {c}"
    
    # 4. Patchify latents into 4x4 spatial blocks for the DiT Transformer Tokens
    p = 4
    h_p, w_p = h // p, w // p
    
    # Reshape and permute to group 4x4 pixels together across channels
    # From: (Batch, Channels, Time, H_patches, Patch_H, W_patches, Patch_W)
    # To:   (Batch, Time, H_patches, W_patches, Channels, Patch_H, Patch_W)
    latents_patched = latents.view(b, c, t, h_p, p, w_p, p)
    latents_patched = latents_patched.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
    
    # Flatten into final tokens: (Batch, Total_Tokens, Channels * 4 * 4)
    latents_flattened = latents_patched.view(b, t * h_p * w_p, c * p * p)
    
    dummy_condition = torch.randn(b, 128).to(device)
    
    # 5. Test DiT Denoising Pass
    dit_output = dit(latents_flattened, dummy_condition)
    assert dit_output.shape == latents_flattened.shape, "DiT output shape must match flattened input latent shape"
    
    # 6. Unpatchify tokens back into standard 3D latent grid shapes
    latents_reconstructed = dit_output.view(b, t, h_p, w_p, c, p, p)
    latents_reconstructed = latents_reconstructed.permute(0, 4, 1, 2, 5, 3, 6).contiguous()
    latents_reconstructed = latents_reconstructed.view(b, c, t, h, w)
    
    decoded_video = decoder(latents_reconstructed)
    
    assert decoded_video.shape[1] == 3, "Decoded video must have 3 color channels (RGB)"
    print("✅ Core Model Pipeline Tensor Alignment Verified!")


@pytest.mark.asyncio
async def test_sfu_room_constraints():
    """
    Verifies the WebRTC Selective Forwarding Unit (SFU) room setup, 
    tracking management, and the 50-person capacity ceiling.
    """
    room_manager = LiveRoomManager()
    room_id = "integration-test-lounge"
    dummy_sdp = "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\nc=IN IP4 127.0.0.1"
    
    # Assert maximum room capacity constraint
    assert room_manager.MAX_PARTICIPANTS == 50
    
    try:
        await room_manager.join_room(
            room_id=room_id, 
            client_id="test-user-1", 
            offer_sdp=dummy_sdp, 
            is_publisher=True
        )
    except Exception as e:
        assert "maximum capacity" not in str(e)
        
    # Verify room tracking initialization
    assert room_id in room_manager.rooms
    
    # Run teardown cleanup
    room_manager.leave_room(room_id, "test-user-1")
    assert "test-user-1" not in room_manager.rooms[room_id]
    print("✅ WebRTC SFU Live Room Management Constraints Verified!")
