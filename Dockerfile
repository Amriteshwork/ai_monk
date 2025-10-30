FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY main.py ./
COPY src/ ./src/

# copy model FROM THE REPO
# this is the path in your GitHub repo:
# runs/detect/vehicle_detection/weights/best.pt
COPY runs/ ./runs/

# dirs for outputs
RUN mkdir -p /app/inference_outputs /app/logs /app/weights

# default: use the model that was copied in the image
ENV MODEL_WEIGHTS_PATH=/app/runs/detect/vehicle_detection/weights/best.pt
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
