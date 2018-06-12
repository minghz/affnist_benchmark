FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

ADD . /app

RUN pip install tqdm

EXPOSE 80
