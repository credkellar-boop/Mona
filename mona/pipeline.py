import torch
import math
import cv2
import numpy as np
import google.generativeai as genai

class MonaLongVideoPipeline:
    """
    Core pipeline for generating long-form video (up to 30 minutes).
    Uses Gemini to direct narrative coherence and audio syncing across chunks.
    """
    def __init__(self, gemini_api_key: str, model_name: str = "mona-v1-base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Configure Gemini as the cognitive orchestrator
        genai.configure(api_key=gemini_api_key)
        self.director = genai.GenerativeModel('gemini-1.5-pro')
        
        print(f"Initialized Mona Pipeline ({self.model_name}) on {self.device}")
        print("Gemini Director Agent connected.")

    @classmethod
    def from_pretrained(cls, model_path: str, gemini_key: str):
        return cls(gemini_api_key=gemini_key, model_name=model_path)

    def generate(self, 
                 prompt: str, 
                 duration_seconds: int = 1800, 
                 fps: int = 24, 
                 resolution: str = "1080p",
                 output_path: str = "output.mp4",
                 enable_streaming: bool = True):
        
        total_frames = duration_seconds * fps
        print(f"Starting generation for: '{prompt}'")
        
        # Ask Gemini to build a structural overview of the 30-minute video
        print("Consulting Gemini for pacing and global narrative structure...")
        master_plan = self.director.generate_content(
            f"You are directing a {duration_seconds} second video. Prompt: {prompt}. "
            "Outline the visual pacing, character consistency rules, and audio design."
        ).text

        if enable_streaming:
            self._generate_in_chunks(total_frames, fps, output_path, prompt, master_plan)
        else:
            print("Warning: Non-streaming generation requires extensive VRAM.")

    def _generate_in_chunks(self, total_frames: int, fps: int, output_path: str, master_prompt: str, master_plan: str):
        chunk_size_frames = 10 * fps  
        total_chunks = math.ceil(total_frames / chunk_size_frames)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (1920, 1080))
        
        memory_state = "Start of video."
