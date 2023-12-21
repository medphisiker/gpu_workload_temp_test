import os

import cv2
import gdown
import ultralytics
from ultralytics import YOLO


if __name__ == "__main__":
    # скачать видео файл с танцами
    video_path = "dance.mp4"
    if not os.path.exists(video_path):
        print("Скачиваем видео с танцами")
        id = "1HtjpDkM-BvBTIlYkU5UDwt0C9KU6z3Uv"
        gdown.download(id=id, output=video_path, quiet=False)

    # изменим на нужный нам
    os.chdir("/workspace")
    print(os.getcwd())  # выведет /workspace

    ultralytics.checks()

    # Создадим TensorRT для сегментации
    yolov8_seg_path = "yolov8l-seg.engine"
    if not os.path.exists(yolov8_seg_path):
        model_seg = YOLO("yolov8l-seg.pt")
        model_seg.export(format="engine", simplify=False)

    # Создадим TensorRT для для оценки ключевых точек
    yolov8_pose_path = "yolov8l-pose.engine"
    if not os.path.exists(yolov8_pose_path):
        model_pose = YOLO("yolov8l-pose.pt")
        model_pose.export(format="engine", simplify=False)

    # загрузим TensorRT-модель для сегментации
    model_seg = YOLO(yolov8_seg_path)

    # загрузим TensorRT-модель для оценки ключевых точек
    model_pose = YOLO(yolov8_pose_path)

    # открываем видео файл
    cap = cv2.VideoCapture(video_path)

    # итерируемся по всем его кадрам
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model_seg.track(frame, persist=True)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imwrite("test.png", annotated_frame)

            break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
