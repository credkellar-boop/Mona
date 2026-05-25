# Mona 🎨🎬

**Mona** is an open-source, long-form video generation framework designed to extend the boundaries of Spatio-Temporal Diffusion Transformers (DiT). While current state-of-the-art models like Sora excel at short, high-fidelity clips, Mona is architected specifically for **long-form narrative coherence**, supporting generations up to **30 minutes** in duration.

---

## 🚀 Key Features

*   **Extended Temporal Coherence:** Utilizes novel sliding-window attention and memory-augmented transformers to prevent narrative drift over long sequences.
*   **Hierarchical Generation:** Generates a low-framerate "keyframe script" first, followed by parallelized temporal upsampling and interpolation chunks.
*   **Infinite-Context VAE:** A streaming Spatio-Temporal VAE that processes video in continuous, overlapping latent blocks to eliminate VRAM bottlenecks.
*   **Audio-Video Co-generation:** Latent-space alignment that syncs audio tracks dynamically over the 30-minute runtime.

---

## 🛠️ Architecture Overview

Generating 30 minutes of video ($1800\text{ seconds}$) at $24\text{ fps}$ requires processing $43,200$ frames. Standard attention mechanisms fail at this scale due to quadratic complexity. Mona solves this via:

1.  **Recurrent Video Transformers (R-DiT):** Passing a compressed "memory state" from previous 1-minute blocks to subsequent blocks.
2.  **Streaming Latent Decoding:** Instead of decoding the entire 30-minute latent space at once, Mona streams chunks to the GPU, decoding and stitching in real-time.

---

## 📦 Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/mona.git](https://github.com/your-username/mona.git)
cd mona
pip install -r requirements.txt
mona/
├── .github/workflows/      # CI/CD pipelines
├── assets/                 # Architecture diagrams, logos, and UI screenshots
├── configs/                # Model and training configuration files (YAML)
├── data/                   # Data preprocessing and loading scripts
├── mona/                   # Main source code package
│   ├── __init__.py
│   ├── models/             # DiT (Diffusion Transformer) & Spatio-Temporal VAE blocks
│   ├── pipeline.py         # Long-video generation pipeline
│   └── utils/              # Memory management, chunking, and streaming utils
├── tests/                  # Unit and integration tests
├── .gitignore
├── LICENSE                 # e.g., Apache 2.0 or MIT
├── README.md               # The main documentation (see below)
└── requirements.txt        # Python dependencies
