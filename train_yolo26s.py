import os
import cv2

# Отключаем конфликты потоков
#cv2.setNumThreads(0)
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"

# МАГИЧЕСКАЯ СТРОКА: Заставляем PyTorch не использовать /dev/shm
#os.environ["PYTORCH_MULTIPROCESSING_SHARING_STRATEGY"] = "file_system"

# Отключаем облачные логи
os.environ["WANDB_MODE"] = "disabled"

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo26s.pt')

    results = model.train(
        data='data_ru.yaml',
        imgsz=640,
        epochs=50,
        patience=20,
        #batch=12,             # Если упадет, снизьте до 4
        name='yolo26s_rtsd_dataset',
        #workers=12,           # Ставим 2 для проверки (это уже ускорит в 2-3 раза)
        #cache=False,         # КРИТИЧНО: не кэшировать датасет в RAM
        amp=True,            # Mixed precision (ускоряет и экономит память)
        optimizer='MuSGD'
        #deterministic=False  # Отключаем жесткую детерминированность
        #resume=True
    )
