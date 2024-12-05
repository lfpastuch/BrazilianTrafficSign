import os
import yaml
import cv2
from ultralytics import YOLO
import random

def detect_image(model, image_path):
    img = cv2.imread(image_path)
    results = model(img)

    # Display the image with class ID
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            class_id = int(box.cls.item())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Inference', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'checkpoints/yolo11_TrafficSign_300eph.pt'))
model = YOLO(weights_path)

image_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset/test/images'))
image_files = sorted(os.listdir(image_dataset))

while True:
    random_image_number = random.randint(1, len(image_files))
    image_path = os.path.join(image_dataset, image_files[random_image_number - 1])
    detect_image(model, image_path)