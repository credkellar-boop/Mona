import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class MonaVideoDataset(Dataset):
    """
    Custom Dataset to load preprocessed video chunks for training the DiT.
    Expects a directory of video frames and an accompanying text caption file.
    """
    def __init__(self, data_root: str, annotation_file: str, sequence_length: int = 240, resolution: int = 512):
        """
        Args:
            data_root: Path to the directory holding folders of video frames.
            annotation_file: Path to a text file containing "folder_name ||| caption"
            sequence_length: Number of continuous frames to sample (e.g., 240 frames = 10s at 24fps)
            resolution: Target spatial size to resize frames.
        """
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.samples = self._load_annotations(annotation_file)
        
        # Spatial transformation matching the VAE input expectations
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Scale to [-1, 1]
        ])

    def _load_annotations(self, filepath: str):
        samples = []
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: Annotation file {filepath} not found. Creating placeholder map.")
            return samples
            
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if "|||" in line:
                    folder, caption = line.strip().split("|||")
                    samples.append({"folder": folder.strip(), "caption": caption.strip()})
        return samples

    def __len__(self):
        return len(self.samples) if self.samples else 10 # Dummy fallback length for empty directories

    def __getitem__(self, idx):
        # Fallback structure if no data exists yet
        if not self.samples:
            dummy_video = torch.zeros((3, self.sequence_length, 512, 512)) # (C, T, H, W)
            return dummy_video, "A placeholder prompt description for debugging."

        sample = self.samples[idx]
        folder_path = os.path.join(self.data_root, sample["folder"])
        all_frames = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))])
        
        # Temporal Chunking / Sampling
        # Ensure we grab a continuous chunk of frames matching sequence_length
        if len(all_frames) >= self.sequence_length:
            start_frame = torch.randint(0, len(all_frames) - self.sequence_length + 1, (1,)).item()
            selected_frames = all_frames[start_frame : start_frame + self.sequence_length]
        else:
            # Pad with the last frame if the clip is shorter than targeted
            selected_frames = all_frames + [all_frames[-1]] * (self.sequence_length - len(all_frames))

        processed_frames = []
        for frame_name in selected_frames:
            img_path = os.path.join(folder_path, frame_name)
            img = Image.open(img_path).convert("RGB")
            processed_frames.append(self.transform(img))

        # Stack frames along the temporal dimension: (Channels, Time, Height, Width)
        video_tensor = torch.stack(processed_frames, dim=1) 
        
        return video_tensor, sample["caption"]
