import torch
import math
import cv2
import numpy as np

class MonaLongVideoPipeline:
    """
    Core pipeline for generating long-form video (up to 30 minutes).
    Uses chunked spatio-temporal decoding to manage VRAM.
    """
    def __init__(self, model_name: str = "mona-v1-base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initialized Mona Pipeline ({self.model_name}) on {self.device}")

    @classmethod
    def from_pretrained(cls, model_path: str):
        return cls(model_name=model_path)

    def generate(self, 
                 prompt: str, 
                 duration_seconds: int = 1800, 
                 fps: int = 24, 
                 resolution: str = "1080p",
                 output_path: str = "output.mp4",
                 enable_streaming: bool = True):
        
        total_frames = duration_seconds * fps
        print(f"Starting generation for: '{prompt}'")
        print(f"Target: {duration_seconds}s | {fps} FPS | Total Frames: {total_frames}")

        if enable_streaming:
            self._generate_in_chunks(total_frames, fps, output_path)
        else:
            print("Warning: Non-streaming generation requires extensive VRAM.")

    def _generate_in_chunks(self, total_frames: int, fps: int, output_path: str):
        chunk_size_frames = 10 * fps  # 10 seconds per chunk
        total_chunks = math.ceil(total_frames / chunk_size_frames)
        
        # Setup OpenCV video writer for continuous appending
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
        
        memory_state = None # Holds context from the previous chunk
        
        for chunk_idx in range(total
