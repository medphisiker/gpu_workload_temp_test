import datetime
import logging
import os
import time

import cv2
import gdown
import GPUtil
import psutil
import ultralytics
from ultralytics import YOLO


def get_temp():
    # берем первую ГПУ
    gpu = GPUtil.getGPUs()[0]
    return gpu.temperature, gpu.load, gpu.memoryUtil


def get_seconds(time_stamp):
    time_obj = datetime.datetime.strptime(time_stamp, "%H:%M:%S")
    time_delta = datetime.timedelta(
        hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second
    )
    total_seconds = time_delta.total_seconds()
    return total_seconds


def setup_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)

    fileHandler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)

    return log


if __name__ == "__main__":
    yolov8_seg = "yolov8l-seg"
    yolov8_pose = "yolov8l-pose"
    load_period = "00:01:01"
    temp_thresh = 80

    load_period = get_seconds(load_period)

    temp_logger = setup_logger("gpu_temp_logger", log_file="gpu_temp.log")
    temp_thresh_logger = setup_logger(
        "gpu_temp_thresh_logger", log_file="gpu_temp_thresh.log"
    )

    # скачать видео файл с танцами
    video_path = "dance.mp4"
    if not os.path.exists(video_path):
        print("Скачиваем видео с танцами")
        
        # короткое видео в 15 секунд для отладки
        id = "1GSLeIBYqMCm_s0ji1yAob0czOQaLy8g_"
        
        # полное видео на 9 минут, для нагрузки
        # id = "1HtjpDkM-BvBTIlYkU5UDwt0C9KU6z3Uv"
        gdown.download(id=id, output=video_path, quiet=False)

    # изменим на нужный нам
    os.chdir("/workspace")
    print(os.getcwd())  # выведет /workspace

    ultralytics.checks()

    # Создадим TensorRT для сегментации
    yolov8_seg_path = f"{yolov8_seg}.engine"
    if not os.path.exists(yolov8_seg_path):
        model_seg = YOLO(f"{yolov8_seg}.pt")
        model_seg.export(format="engine", simplify=False)

    # Создадим TensorRT для для оценки ключевых точек
    yolov8_pose_path = f"{yolov8_pose}.engine"
    if not os.path.exists(yolov8_pose_path):
        model_pose = YOLO(f"{yolov8_pose}.pt")
        model_pose.export(format="engine", simplify=False)

    # загрузим TensorRT-модель для сегментации
    model_seg = YOLO(yolov8_seg_path)

    # загрузим TensorRT-модель для оценки ключевых точек
    model_pose = YOLO(yolov8_pose_path)

    start_time = time.time()
    working_time = 0
    stop_flag = False
    
    while working_time < load_period and not stop_flag:
        # открываем видео файл
        cap = cv2.VideoCapture(video_path)

        # итерируемся по всем его кадрам
        frame_num = 0
        while cap.isOpened() and not stop_flag:
            success, frame = cap.read()

            if success:
                results = model_seg.track(frame, persist=True)

                # отрисовываем результаты сегментации
                frame = results[0].plot()

                results = model_pose.predict(frame)

                # отрисовываем результаты оценки позы
                frame = results[0].plot()

                gpu_temp, gpu_load, gpu_memory = get_temp()
                print(gpu_temp, gpu_load, gpu_memory)
                # сохранить кадр в виде картинки
                # cv2.imwrite("test.png", frame)

                if frame_num % 8 == 0:
                    temp_logger.info(
                        f"gpu temp: {gpu_temp} gpu load: {gpu_load} "
                        f"gpu mem: {gpu_memory}"
                    )

                if temp_thresh < gpu_temp:
                    temp_thresh_logger.info(
                        f"The temperature threshold {temp_thresh}C has been exceeded!"
                    )
                    temp_thresh_logger.info(
                        f"gpu temp: {gpu_temp} gpu load: {gpu_load} "
                        f"gpu mem: {gpu_memory}"
                    )

                    stop_flag = True

                frame_num += 1
            else:
                break

        end_time = time.time()
        working_time = end_time - start_time

    cap.release()
    cv2.destroyAllWindows()
