from ultralytics import YOLO
import torch

model = YOLO("yolov8s.pt")
model.tune(
    data='data_ru.yaml',
    batch=32, 
    epochs=25, 
    iterations=5,
    optimizer='AdamW',
    plots=True,
    save=True,
    val=False,
    device='cuda'
)
