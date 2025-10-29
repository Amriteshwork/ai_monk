from ultralytics import YOLO
from src.config import (
    DATA_YAML_PATH,
    LOGGER,
    YOLO_EPOCHS,
    YOLO_IMAGE_SIZE,
    YOLO_BATCH,
    YOLO_WORKERS,
)

from src.data_split import split_yolo_dataset
from src.config import DATASET_ROOT


def main():
    model = YOLO(model='yolo11s.pt')

    results = model.train(
        data=str(DATA_YAML_PATH),
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMAGE_SIZE, 
        batch=YOLO_BATCH,
        workers=YOLO_WORKERS,
        optimizer="adamw",
        patience=20,
        name="vehicle_detection",
        project="runs/detect",
    )

    LOGGER.info("Training done.")
    LOGGER.info("Best weights: %s", results.save_dir + "/weights/best.pt")

if __name__ == "__main__":
    import os
import shutil
import random
from pathlib import Path
import yaml

def split_yolo_dataset(
    dataset_root,
    train_ratio=0.8,
    image_exts=(".jpg", ".jpeg", ".png"),
):
    dataset_root = Path(dataset_root)
    img_dir = dataset_root / "train" / "images"
    lbl_dir = dataset_root / "train" / "labels"

    val_dir = dataset_root / "valid"
    val_img_dir = val_dir / "images"
    val_lbl_dir = val_dir / "labels"
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image paths
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in image_exts]
    random.shuffle(images)
    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = n_total - n_train

    print(f"Total images: {n_total} -> Train: {n_train}, Val: {n_val}")

    val_imgs = images[n_train:]

    # Move/copy validation samples
    for img_path in val_imgs:
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        # Move image
        shutil.move(str(img_path), val_img_dir / img_path.name)
        # Move label if exists
        if lbl_path.exists():
            shutil.move(str(lbl_path), val_lbl_dir / lbl_path.name)

    # Rewrite data.yaml
    yaml_path = dataset_root / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    else:
        data = {}

    data["train"] = "train/images"
    data["val"] = "valid/images"
    if "test" in data:
        data.pop("test", None)

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print("✅ Split complete and data.yaml updated.")
    print(f"Updated data.yaml:\n{yaml.dump(data, sort_keys=False)}")


if __name__ == "__main__":

    split_yolo_dataset(dataset_root=DATASET_ROOT, train_ratio=0.8)
    main()
