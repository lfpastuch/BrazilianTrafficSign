import os
import yaml
from ultralytics import YOLO
from datetime import datetime

# Get dataset configuration
data_config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/data.yaml'))
with open(data_config_file, 'r') as file:
    data_config = yaml.safe_load(file)

# Initialize YOLO model
model = YOLO(os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/yolo11.yaml')))

# Print dataset configuration
print("Dataset Configuration:")
print(data_config)

# Train the model
print("Starting training...")
results = model.train(
    data=data_config_file,
    device='cuda',
    epochs=300,
    imgsz=640,
    batch=16,
    name='yolov8_traffic_signs'
)

print("Training completed.")
print("Results:")
print(results)