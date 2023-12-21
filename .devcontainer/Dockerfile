# образ для контейнера
FROM ultralytics/ultralytics:8.0.228

# отключаем интерактивные вопросы при установке пакетов
# при создании docker-контейнера
ARG DEBIAN_FRONTEND=noninteractive

# установим opencv-python
# для этого нам понадобится ее зависимость libgl1
# см. https://stackoverflow.com/a/68666500
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --no-cache opencv-python

# установим другие нужные библиотеки
RUN pip install --no-cache onnxruntime-gpu==1.16.3 gdown==4.7.1 GPUtil==1.4.0

# установим kernel для Jupyter Notebook
RUN pip install --no-cache ipykernel==6.23.0

# чистим систему от лишнего
RUN apt-get clean && apt-get autoremove

# папка для подключения репозитория с кодом
WORKDIR /workspace

COPY ./yolov8 /workspace

# ENTRYPOINT ["python", "run.py"]