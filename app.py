from potassium import Potassium, Request, Response
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

MODEL_NAME_OR_PATH = "damo-vilab/text-to-video-ms-1.7b"

app = Potassium("text-to-video-ms-1.7b")

@app.init
def init() -> dict:
    """Initialize the application with the model and processor."""
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME_OR_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME_OR_PATH)
    return {
        "model": model,
        "processor": processor
    }
    
@app.handler()
def handler(context: dict, request: Request) -> Response:
    """Handle a request to generate text from the image."""
    model = context.get("model")
    processor = context.get("processor")
    image_b64 = request.json.get("image")
    image_bytes = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_bytes))
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    output = model.config.id2label[predicted_class_idx]
    return Response(json={"output": output}, status=200)

if __name__ == "__main__":
    app.serve()