from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data='data_ru.yaml',
    imgsz=640,
    epochs=25,
    batch=48,
    name='yolov8s_rtsd_dataset',
)
