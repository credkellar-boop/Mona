# Mona

<div align="center">

[![Mona CI](https://img.shields.io/github/actions/workflow/status/credkellar-boop/mona/ci.yml?branch=main&logo=github-actions&logoColor=white&style=flat-square)](https://github.com/credkellar-boop/mona/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?logo=python&logoColor=white&style=flat-square)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D%202.2.0-ee4c2c?logo=pytorch&logoColor=white&style=flat-square)](https://pytorch.org)
[![Gemini](https://img.shields.io/badge/Gemini%20AI-Orchestration-blueviolet?logo=googlegemini&logoColor=white&style=flat-square)](https://deepmind.google/technologies/gemini/)
[![WebRTC SFU](https://img.shields.io/badge/WebRTC-SFU%20Enabled-green?logo=webrtc&logoColor=white&style=flat-square)](https://github.com/aiortc/aiortc)
[![License](https://img.shields.io/github/license/credkellar-boop/mona?color=orange&style=flat-square)](LICENSE)

An open-source, long-form video generation framework capable of producing up to 30 minutes of continuous, temporally coherent video using Spatio-Temporal Diffusion Transformers (DiT), 3D Causal VAE architecture, and Gemini-driven narrative orchestration. It also features a real-time, low-latency live streaming and 50-person group video chat server.

[Features](#features) • [Project Structure](#-project-structure) • [Installation](#installation) • [Getting Started](#getting-started) • [Production Deployment](#production-deployment)

</div>

---

## Features

- **Long-Form Video Generation:** Generates up to 30 minutes of continuous 1080p video using sequential temporal chunking.
- **Gemini Director Agent:** Leverages Gemini to map narrative pacing, manage memory context across chunks, and maintain character/asset consistency.
- **Spatio-Temporal DiT:** A state-of-the-art Diffusion Transformer model designed to capture both spatial geometry and smooth temporal motions.
- **3D Causal VAE:** Advanced spatio-temporal video compression that ensures past video frames cannot leak into future frames, saving local VRAM.
- **Automated Dataset Engine:** Auto-captions raw training videos via the Gemini File API and cuts clips automatically.
- **Real-Time Video Stack:** Includes a Selective Forwarding Unit (SFU) WebRTC signaling node capable of hosting 1-on-1 chats and up to 50-person group video rooms.

---

## 📂 Project Structure

The repository is organized following clean Python software development practices:

```text
mona/
├── .github/
│   └── workflows/
│       └── ci.yml             # Continuous Integration automated build pipelines
├── configs/
│   └── default.yaml           # Model hyperparameters and hardware configurations
├── data/
│   ├── train_frames/          # Local preprocessed training video clips (git-ignored)
│   ├── annotations.txt        # Training frame-to-text annotation mappings
│   ├── dataset.py             # PyTorch Custom Dataset & temporal loading modules
│   └── auto_captioner.py      # Automated frame slicer and Gemini pipeline captioner
├── mona/
│   ├── __init__.py            # Exposes package initializations
│   ├── pipeline.py            # Main generation pipeline & sliding-window context engine
│   ├── models/
│   │   ├── __init__.py        # Exposes DiT and VAE architectures
│   │   ├── dit.py             # Spatio-Temporal Diffusion Transformer (adaLN modulation)
│   │   └── vae.py             # 3D Causal Variational Autoencoder with Media Relays
│   └── utils/
│       ├── streaming.py       # Context/VRAM offloading utility banks
│       └── live_sfu.py        # Asynchronous WebRTC SFU multiparty packet router
├── live_server.py             # Aiohttp Signaling endpoint server for real-time video
├── main.py                    # Root entry point for standard video inference runs
├── train.py                   # Diffusion transformer training framework loop execution
├── requirements.txt           # Unified external library package configurations
└── README.md                  # System overview and deployment documentation
