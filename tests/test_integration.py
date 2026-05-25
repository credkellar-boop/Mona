# In tests/test_integration.py

# ... (Previous VAE encoding step)
latents = encoder(dummy_video)
b, c, t, h, w = latents.shape

# Patch size (p) must be defined as used in your DiT model (likely 4x4)
p = 4 
h_p, w_p = h // p, w // p

# 1. Reshape to separate patches: (Batch, Channels, Time, H_p, P, W_p, P)
latents_patched = latents.view(b, c, t, h_p, p, w_p, p)

# 2. Permute to bring patch dimensions together: (Batch, Time, H_p, W_p, Channels, P, P)
latents_patched = latents_patched.permute(0, 2, 3, 5, 1, 4, 6).contiguous()

# 3. Flatten correctly: (Batch, Time * H_p * W_p, Channels * P * P)
# This results in the specific feature dimension the DiT expects
latents_flattened = latents_patched.view(b, t * h_p * w_p, c * p * p)

# Pass to DiT
dummy_condition = torch.randn(b, 128).to(device)
dit_output = dit(latents_flattened, dummy_condition)
