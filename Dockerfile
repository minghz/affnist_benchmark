FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

ADD . /app

RUN pip install --upgrade pip
RUN pip install tqdm numpy scipy

EXPOSE 80
