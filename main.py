import io
import uuid
import numpy as np
from typing import List

import uvicorn
import gradio as gr
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image

from src.config import MODEL_WEIGHTS_PATH, ID_TO_NAME, LOGGER, OUTPUT_DIR

# Pydantic Models for API Response 

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    box: BoundingBox

class InferenceResponse(BaseModel):
    detections: List[DetectionResult]
    annotated_image_path: str

# FastAPI Application Setup
app = FastAPI(
    title="Vehicle Detection API",
    description="An AI service using a YOLOv11s model to detect vehicles.",
    version="1.0.0",
)

# Model Loading
try:
    model = YOLO(MODEL_WEIGHTS_PATH)
    LOGGER.info(f"Successfully loaded model from {MODEL_WEIGHTS_PATH}")
except Exception as e:
    LOGGER.error(f"Failed to load model from {MODEL_WEIGHTS_PATH}. Error: {e}")
    raise RuntimeError(f"Could not load YOLO model: {e}")

# Core Detection Logic
def run_detection(input_image: Image.Image):
    """
    Runs the YOLO model on a PIL Image.
    
    Args:
        input_image: A PIL.Image.Image object.
        
    Returns:
        Tuple[np.ndarray, List[DetectionResult]]:
            - The annotated image as an RGB NumPy array.
            - A list of DetectionResult pydantic models.
    """
    # Run model inference
    results = model(input_image)
    result = results[0]

    # Get annotated image (as an RGB numpy array)
    annotated_image_array = result.plot() 
    
    # Process detection results
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = box.conf[0].item()
        coords = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]

        bounding_box = BoundingBox(
            x_min=coords[0],
            y_min=coords[1],
            x_max=coords[2],
            y_max=coords[3]
        )
        
        detection_result = DetectionResult(
            class_name=ID_TO_NAME.get(class_id, "unknown"),
            confidence=confidence,
            box=bounding_box
        )
        detections.append(detection_result)
        
    return annotated_image_array, detections

# AI Backend Service (API Endpoint)
@app.post("/detect/", response_model=InferenceResponse)
async def detect_objects(file: UploadFile = File(...)):
    """
    The API endpoint for object detection.
    """
    if not file.content_type.startswith("image/"):
        LOGGER.warning(f"Invalid file type uploaded: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # 1. Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Call the core detection logic
        annotated_image_array, detections = run_detection(image)

        # Save annotated image
        save_filename = f"{uuid.uuid4()}.jpg"
        save_path = OUTPUT_DIR / save_filename
        Image.fromarray(annotated_image_array).save(save_path) # Save the annotated image
        LOGGER.info(f"Saved annotated image to: {save_path}")

        LOGGER.info(f"Detected {len(detections)} objects in {file.filename} via API")

        # 5. Return the JSON response
        return InferenceResponse(
            detections=detections,
            annotated_image_path=str(save_path)
        )

    except Exception as e:
        LOGGER.error(f"Error during API inference: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "An internal server error occurred during inference.", "error": str(e)}
        )

# 2. UI End (Gradio Interface)
def gradio_inference_fn(image_numpy: np.ndarray):
    """
    A wrapper function for our core logic to fit Gradio's I/O.
    
    Args:
        image_numpy: Input image as a NumPy array (from gr.Image).
    
    Returns:
        Tuple[np.ndarray, dict]:
            - The annotated image as a NumPy array.
            - The detection results as a JSON-friendly dict.
    """
    # Convert numpy array to PIL Image
    image_pil = Image.fromarray(image_numpy)
    
    # Call the core detection logic
    annotated_image_array, detections_pydantic = run_detection(image_pil)

    # Convert pydantic objects to plain dicts for Gradio's JSON output
    detections_json = [d.model_dump() for d in detections_pydantic]
    
    return annotated_image_array, detections_json

# Define the Gradio interface
gradio_ui = gr.Interface(
    fn=gradio_inference_fn,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(type="numpy", label="Annotated Image"),
        gr.JSON(label="Detection Results")
    ],
    title="Vehicle Detection",
    description="Upload an image to detect vehicles."
)

# Mount the Gradio app onto the FastAPI app
app = gr.mount_gradio_app(app, gradio_ui, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

