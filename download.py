import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

MODEL_NAME_OR_PATH = "google/vit-base-patch16-224"

def download_model() -> tuple:
    """Create the pipe."""
    pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
    return pipe

if __name__ == "__main__":
    download_model()
    