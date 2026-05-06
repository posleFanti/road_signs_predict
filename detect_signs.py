from ultralytics import YOLO
import cv2 as cv

model = YOLO('yolov8s.pt')

cap = cv.VideoCapture('/Users/alex/dashcam_videos/NO20260415-165809-003846F.MP4')

prev_time = 0
frame_count = 0
PROCESS_EVERY_N_FRAMES = 3  # Можно легко изменить на 3, 4, 5 и т.д.

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    # Обрабатываем только каждый N-й кадр
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Детекция
        results = model(frame, conf=0.5, iou=0.45)
        
        # Получение результатов
        annotated_frame = results[0].plot()
            
        # Вывод информации о детекциях
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            print(f"Обнаружен: {label}, уверенность: {conf:.2f}")
        
        cv.imshow('YOLOv8 Detection', annotated_frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
