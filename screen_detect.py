from ultralytics import YOLO
import cv2
import mss
import numpy as np
import time
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=r"runs\detect\runs\road_signs_yolo26l_1280_b12-18\weights\best.pt",
        help="Путь к модели best.pt или last.pt",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Порог уверенности. Ниже = больше находок, но больше мусора.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Размер входа YOLO. 640 быстрее, 736/800 лучше для мелких объектов.",
    )

    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Номер монитора. Обычно 1 = основной экран.",
    )

    parser.add_argument(
        "--print",
        action="store_true",
        help="Печатать найденные объекты в терминал.",
    )

    args = parser.parse_args()

    print(f"Загружаю модель: {args.model}")
    model = YOLO(args.model)

    print("Старт захвата экрана.")
    print("Нажми Q в окне с картинкой, чтобы выйти.")

    frame_count = 0
    fps = 0.0
    fps_timer = time.time()
    last_terminal_print = 0.0

    with mss.MSS() as sct:
        monitor = sct.monitors[args.monitor]

        print(f"Захватываю монитор #{args.monitor}: {monitor}")

        while True:
            # Захват экрана
            screenshot = sct.grab(monitor)

            # MSS отдаёт BGRA, переводим в BGR для OpenCV
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Прогон через YOLO
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device='mps',
                verbose=False,
            )

            result = results[0]

            # Рисуем боксы на кадре
            annotated = np.ascontiguousarray(result.plot()).copy()

            # Считаем FPS
            frame_count += 1
            now = time.time()

            if now - fps_timer >= 1.0:
                fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now

            # Пишем FPS на картинку
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f} | conf: {args.conf} | imgsz: {args.imgsz}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Печать в терминал не каждый кадр, чтобы консоль не превратилась в канализацию
            if args.print and now - last_terminal_print > 0.5:
                last_terminal_print = now

                boxes = result.boxes

                if boxes is not None and len(boxes) > 0:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Найдено объектов: {len(boxes)}")

                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        name = model.names.get(cls_id, str(cls_id))

                        print(
                            f"  class={cls_id} name={name} conf={conf:.2f} "
                            f"box=({x1},{y1},{x2},{y2})"
                        )

            # Показываем окно
            cv2.imshow("YOLO screen detection - press Q to quit", annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
