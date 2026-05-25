import os
from mona.pipeline import MonaLongVideoPipeline

# It's best practice to keep API keys in environment variables, 
# but you can paste yours directly here for local testing.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

if __name__ == "__main__":
    # Initialize your pipeline
    pipeline = MonaLongVideoPipeline(gemini_api_key=GEMINI_API_KEY)

    # Let's start with a 60-second test before trying a full 30 minutes!
    pipeline.generate(
        prompt=(
            "A seamless cinematic tracking shot following a glowing blue butterfly "
            "as it flies through a dense, neon-lit cyberpunk city, navigating "
            "traffic and alleyways."
        ),
        duration_seconds=60, 
        fps=24,
        resolution="1080p",
        output_path="mona_test_render.mp4"
    )
