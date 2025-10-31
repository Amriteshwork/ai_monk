# AI Monk — Vehicle Detection using YOLO

**Vehicle Detection** application built with **YOLOv11**, **FastAPI**, and **Gradio**.  
It provides an interactive web interface and REST API for running inference on images using a pretrained model.
It can predict: **bus, car, microbus, motorbike, pickup-van, and truck**

---

## Project Structure

```
ai_monk/
├── data/                          # raw training data (not required for inference)
├── runs/detect/vehicle_detection/ # trained YOLO model and logs
│   └── weights/
│       ├── best.pt                # pretrained weights (used for inference)
│       └── last.pt
├── src/
|   └── __init__.py                           
│   └── config.py                  # training utilities, configs
|   └── data_split.py
├── test_image/                    # sample test images
├── main.py                        # FastAPI + Gradio app for inference
├── train.py                       # training entrypoint
├── test_detection.py               # testing / evaluation script
├── requirements.txt
└── Dockerfile
```

---

## Quick Start (Docker)

You can run this project **without installing Python or dependencies** — using Docker.

### 1 Clone the repository

```bash
git clone https://github.com/Amriteshwork/ai_monk.git
cd ai_monk
```

### 2 Build the Docker image

```bash
docker build -t vehicle-detector .
```

### 3 Run the container

Run the container and expose port **8000**:

```bash
docker run -p 8000:8000 vehicle-detector
```

Now open your browser at:  **http://localhost:8000**

You’ll see the **Gradio interface** for uploading images and viewing detection results.

---

## Optional: Persistent outputs

To save the detected images on your host machine:

```bash
docker run \
  -p 8000:8000 \
  -v $(pwd)/inference_outputs:/app/inference_outputs \
  vehicle-detector
```

Detected images will appear in your local `inference_outputs/` folder.

---

## Optional: Custom model weights

If you want to run inference with your own trained weights, mount them and override the environment variable:

```bash
docker run \
  -p 8000:8000 \
  -v $(pwd)/my_weights:/app/weights \
  -e MODEL_WEIGHTS_PATH=/app/weights/best.pt \
  vehicle-detector
```

---


## Development Setup (without Docker)

If you prefer running locally:

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py
```

Then open [http://localhost:8000](http://localhost:8000)

---

## Configuration

The configuration is defined in `src/config.py`:

```python
MODEL_WEIGHTS_PATH = Path(os.getenv(
    "MODEL_WEIGHTS_PATH",
    f"{BASE_DIR}/runs/detect/vehicle_detection/weights/best.pt"
))
```

- By default, it loads the pretrained model from `runs/detect/vehicle_detection/weights/best.pt`
- You can override this path using the environment variable `MODEL_WEIGHTS_PATH`

---

## API Endpoint

Once running, you can also use the FastAPI endpoint:

**POST /detect/**  
Upload an image file:

```bash
curl -X POST "http://localhost:8000/detect/" \
  -F "file=@test_image/test_car_3.jpg"
```


## 👨‍💻 Author

**Amritesh Work**  
GitHub: [@Amriteshwork](https://github.com/Amriteshwork)

---