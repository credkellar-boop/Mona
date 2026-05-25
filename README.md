# Mona

![GitHub Actions CI](https://img.shields.io/github/actions/workflow/status/credkellar-boop/mona/ci.yml?branch=main&style=flat-square&logo=github-actions&logoColor=white)
![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D%202.2.0-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Gemini AI](https://img.shields.io/badge/Gemini%20AI-Orchestration-blueviolet?style=flat-square&logo=googlegemini&logoColor=white)
![WebRTC SFU](https://img.shields.io/badge/WebRTC-SFU%20Enabled-green?style=flat-square&logo=webrtc&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Supported-0db7ed?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/github/license/credkellar-boop/mona?style=flat-square&color=orange)

An open-source, ultra-high-definition video generation framework capable of producing up to 30 minutes of continuous, temporally coherent video at 4K Cinematic Resolution. 

---

## 🚀 Key Architectural Modules

* **Spatio-Temporal DiT:** Scalable transformer architecture modeling joint spatial layout and long-range temporal consistency.
* **Stream-Ready WebRTC SFU:** Integrated live-streaming server capabilities built using `aiortc` for real-time video streaming processing pipelines.
* **Gemini AI Orchestration:** Multi-modal narrative prompt conditioning to orchestrate deep video semantic structure over long periods.
* **Containerized Training/Inference Worker:** Production-grade deployment support with NVIDIA Container Toolkit configurations out of the box.

---

## 📂 Project Structure

```text
Mona/
├── .github/
│   └── workflows/
│       └── ci.yml                # Automated Python & System dependency CI pipeline
├── configs/
│   └── default.yaml              # Hyperparameter settings and model layout definitions
├── data/
│   ├── auto_captioner.py         # Automated metadata generation using multimodal vision models
│   └── dataset.py                # Highly optimized video frame tokenization data loader
├── mona/
│   ├── __init__.py               # Core package initialization
│   ├── models/
│   │   ├── __init__.py           # Model sub-package module resolution
│   │   └── ...                   # SpatioTemporalVAE and DiT network architectures
│   ├── utils/
│   │   └── ...                   # Computational helpers and tensor geometric transforms
│   └── pipeline.py               # Frame-generation synthesis loop engine
├── tests/
│   └── test_integration.py       # Frame dimensions & integration pipeline assertion suite
├── .env.example                  # Environmental variables context placeholder
├── .gitignore                    # Local storage and cache filtering configurations
├── Dockerfile                    # Isolated runtime container specification for CUDA systems
├── docker-compose.yml            # Multi-node structure orchestration (Compute Worker & SFU Streaming Node)
├── LICENSE                       # Repository open-source permissions file
├── live_server.py                # WebRTC media connection streaming node
├── main.py                       # Operational entry point for localized generation runs
├── requirements.txt              # Complete Python dependency configuration
└── train.py                      # Multi-GPU spatial-temporal training runtime script
