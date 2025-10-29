from ultralytics import YOLO

model = YOLO("/media/amritesh/bytesviewhdd1/ai_stuff/ai_monk_lab/runs/detect/pothole_car_detector/weights/best.pt") 

results = model("/media/amritesh/bytesviewhdd1/ai_stuff/ai_monk_lab/test_image/test_potholes_2.jpg")
results[0].show() 