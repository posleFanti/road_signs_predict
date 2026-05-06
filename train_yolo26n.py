from ultralytics import YOLO

model = YOLO('yolo26n.pt')

results = model.train(
    data='data_ru.yaml',
    imgsz=640,
    epochs=25,
    batch=48,
    name='yolo26n_rtsd_dataset',
)
