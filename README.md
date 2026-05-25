# Mona

<div align="center">

[![Mona CI](https://img.shields.io/github/actions/workflow/status/credkellar-boop/mona/ci.yml?branch=main&logo=github-actions&logoColor=white&style=flat-square)](https://github.com/credkellar-boop/mona/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?logo=python&logoColor=white&style=flat-square)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D%202.2.0-ee4c2c?logo=pytorch&logoColor=white&style=flat-square)](https://pytorch.org)
[![Gemini](https://img.shields.io/badge/Gemini%20AI-Orchestration-blueviolet?logo=googlegemini&logoColor=white&style=flat-square)](https://deepmind.google/technologies/gemini/)
[![WebRTC SFU](https://img.shields.io/badge/WebRTC-SFU%20Enabled-green?logo=webrtc&logoColor=white&style=flat-square)](https://github.com/aiortc/aiortc)
[![License](https://img.shields.io/github/license/credkellar-boop/mona?color=orange&style=flat-square)](LICENSE)


An open-source, ultra-high-definition video generation framework capable of producing up to 30 minutes of continuous, temporally coherent video at **8K Cinematic Resolution ($7680 \times 4320$)**. Mona achieves this using a hierarchical combination of Spatio-Temporal Diffusion Transformers (DiT), 3D Causal VAE architectures, Gemini-driven narrative orchestration, and tile-based spatial super-resolution.

[Features](#features) • [Project Structure](#-project-structure) • [Installation](#installation) • [Getting Started](#getting-started) • [Production Deployment](#production-deployment)

</div>

---

## Features

- **8K Cinematic Generation:** Generates up to 30 minutes of video scaled to true 8K ($7680 \times 4320$) at cinematic frame rates.
- **Hierarchical Latent Upscaling:** To circumvent hardware VRAM limitations, the core architecture generates structural baselines at a lower latent resolution, passing frames through a tiled spatial upscaling network.
- **Gemini Director Agent:** Leverages Gemini to map long-form narrative pacing, manage memory context states across sequential 10-second chunks, and maintain rigid character/environmental consistency.
- **Spatio-Temporal DiT:** A Diffusion Transformer model utilizing modulated Adaptive LayerNorm (adaLN) to handle high-fidelity spatial details and continuous temporal motion.
- **3D Causal VAE:** Advanced video token compression that enforces strict causal dependencies along the timeline (the past cannot see the future), eliminating frame bleed.
- **Real-Time Video Stack:** Includes a built-in Selective Forwarding Unit (SFU) WebRTC signaling node capable of hosting 1-on-1 chats and up to 50-person group video streams.

---

## 📂 Project Structure

The repository layout organizes the generation engine and the high-resolution upscaling stack cleanly:

```text
mona/
├── .github/
│   └── workflows/
│       └── ci.yml             # Continuous Integration automated build pipelines
├── configs/
│   └── default.yaml           # 8K hyperparameters and hardware tile configurations
├── data/
│   ├── train_frames/          # Local preprocessed training video clips (git-ignored)
│   ├── annotations.txt        # Training frame-to-text annotation mappings
│   ├── dataset.py             # PyTorch Custom Dataset & temporal loading modules
│   └── auto_captioner.py      # Automated frame slicer and Gemini pipeline captioner
├── mona/
│   ├── __init__.py            # Exposes package initializations
│   ├── pipeline.py            # Main 8K generation pipeline & sliding-window context engine
│   ├── models/
│   │   ├── __init__.py        # Exposes DiT, VAE, and Upscaler architectures
│   │   ├── dit.py             # Spatio-Temporal Diffusion Transformer
│   │   ├── vae.py             # 3D Causal Variational Autoencoder
│   │   └── upscaler.py        # Tiled Spatial Super-Resolution 8K Network
│   └── utils/
│       ├── streaming.py       # Context/VRAM offloading utility banks
│       └── live_sfu.py        # Asynchronous WebRTC SFU multiparty packet router
├── live_server.py             # Aiohttp Signaling endpoint server for real-time video
├── main.py                    # Root entry point for standard 8K video inference runs
├── train.py                   # Diffusion transformer training framework loop execution
├── requirements.txt           # Unified external library package configurations
└── README.md                  # System overview and deployment documentation
