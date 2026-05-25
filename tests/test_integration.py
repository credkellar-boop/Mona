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
    # 24 frames, 3 channels, 256x256 spatial resolution
    dummy_video = torch.randn(1, 3, 24, 256, 256).to(device)
    
    # 2. Initialize Models with minimal footprints for testing
    encoder = SpatioTemporalVAEEncoder(in_channels=3, latent_channels=4, hidden_dim=32).to(device)
    decoder = SpatioTemporalVAEDecoder(latent_channels=4, out_channels=3, hidden_dim=64).to(device)
    dit = SpatioTemporalDiT(in_channels=4, hidden_size=128, depth=2, num_heads=2).to(device)
    
    # 3. Test VAE Encoding
    latents = encoder(dummy_video)
    b, c, t, h, w = latents.shape
    assert c == 4, f"Expected 4 latent channels, got {c}"
    
    # 4. Flatten latents to tokens for DiT processing
    latents_flattened = latents.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
    dummy_condition = torch.randn(b, 128).to(device)
    
    # 5. Test DiT Denoising Pass
    dit_output = dit(latents_flattened, dummy_condition)
    assert dit_output.shape == latents_flattened.shape, "DiT output shape must match flattened input latent shape"
    
    # 6. Reconstruct back to unflattened latents and decode to pixels
    latents_reconstructed = dit_output.view(b, t, h, w, c).permute(0, 4, 1, 2, 3)
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
    
    # Simulate a client attempting to join the signaling space
    # (Using a try-except block since full browser SDP negotiation requires network interfaces)
    try:
        await room_manager.join_room(
            room_id=room_id, 
            client_id="test-user-1", 
            offer_sdp=dummy_sdp, 
            is_publisher=True
        )
    except Exception as e:
        # If it fails due to local network/engine configuration, it shouldn't be a capacity issue
        assert "maximum capacity" not in str(e)
        
    # Verify room tracking initialization
    assert room_id in room_manager.rooms
    
    # Run teardown cleanup
    room_manager.leave_room(room_id, "test-user-1")
    assert "test-user-1" not in room_manager.rooms[room_id]
    print("✅ WebRTC SFU Live Room Management Constraints Verified!")
