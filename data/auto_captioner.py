import os
import time
import cv2
import google.generativeai as genai

class MonaAutoCaptioner:
    """
    Automates training data creation by slicing local video files into frames
    and using Gemini to generate highly structured caption annotations.
    """
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        # Using gemini-1.5-flash for fast, cost-effective dataset annotation
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        print("Mona Auto-Captioner initialized with Gemini Flash.")

    def process_video_dataset(self, source_video_path: str, output_dir: str, annotation_file_path: str):
        """
        Processes a full video file: splits it into physical folders of frames
        and requests a descriptive training prompt from Gemini.
        """
        video_name = os.path.splitext(os.path.basename(source_video_path))[0]
        target_frame_folder = os.path.join(output_dir, video_name)
        os.makedirs(target_frame_folder, exist_ok=True)

        print(f"\n1. Slicing '{source_video_path}' into frames...")
        self._extract_frames(source_video_path, target_frame_folder)

        print(f"2. Uploading video context to Gemini File API...")
        gemini_file = genai.upload_file(path=source_video_path)
        
        # Wait for Gemini infrastructure to process the video frames
        while gemini_file.state.name == "PROCESSING":
            print("   Waiting for Gemini to finish processing video tensors...")
            time.sleep(7)
            gemini_file = genai.get_file(gemini_file.name)

        if gemini_file.state.name == "FAILED":
            raise ValueError("Gemini video ingestion failed.")

        print("3. Querying Gemini for training-optimized diffusion prompts...")
        prompt = (
            "Analyze this video and provide a single, detailed, highly descriptive sentence "
            "summarizing the actions, lightning, characters, and cinematic style. "
            "Optimize it for text-to-video diffusion model training. Do not include prefixes."
        )
        
        response = self.model.generate_content([prompt, gemini_file])
        caption = response.text.strip()
        print(f"✨ Generated Caption: {caption}")

        # Append to dataset annotations mapping file
        print(f"4. Updating {annotation_file_path}...")
        with open(annotation_file_path, "a", encoding="utf-8") as f:
            f.write(f"{video_name} ||| {caption}\n")

        # Clean up the cloud file memory footprint
        genai.delete_file(gemini_file.name)
        print("Done processing sample.")

    def _extract_frames(self, video_path: str, output_folder: str):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Save frames sequentially matching dataset module expectations
            frame_path = os.path.join(output_folder, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        cap.release()
        print(f"   Successfully extracted {frame_idx} frames to {output_folder}")

if __name__ == "__main__":
    # Local quick-test execution block
    import os
    API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
    
    captioner = MonaAutoCaptioner(gemini_api_key=API_KEY)
    
    # Place any sample video (like an mp4 clip) in your root to seed your training setup
    # captioner.process_video_dataset("sample_clip.mp4", "data/train_frames", "data/annotations.txt")
