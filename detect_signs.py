from ultralytics import YOLO
import cv2 as cv
import os

model = YOLO('runs/detect/road_signs_yolo26l_1280_b12-28/weights/best.pt')

filename = 'NO20260420-073101-003926F.MP4'
cap = cv.VideoCapture(f'/Users/alex/dashcam_videos/{filename}')

prev_time = 0
frame_count = 0
PROCESS_EVERY_N_FRAMES = 3  # Можно легко изменить на 3, 4, 5 и т.д.
prev_boxes = None

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    frame_count += 1
    
    # Обрабатываем только каждый N-й кадр
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Детекция
        results = model(frame, conf=0.5, iou=0.45, device='mps', verbose=False)
        
        # Получение результатов
        annotated_frame = results[0].plot(line_width=5, font_size=30)
            
        # Вывод информации о детекциях
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            print(f"Обнаружен: {label}, уверенность: {conf:.2f}")

        cv.imshow('YOLO Detection', annotated_frame)
        annotated_frame = cv.resize(annotated_frame, (1600, 900))
        
        os.makedirs(f'./detection_images/YOLO26n1024/{filename}_frames', exist_ok=True)
        
        save_path = f'./detection_images/YOLO26n1024/{filename}_frames/{filename}_frame_{frame_count}.jpg'

        if not os.path.isfile(save_path) and results[0].boxes and results[0].boxes != prev_boxes:
            cv.imwrite(save_path, annotated_frame)
            print(f"Сохранил кадр: {save_path}")
        prev_boxes = results[0].boxes

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
