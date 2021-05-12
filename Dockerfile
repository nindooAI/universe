FROM tensorflow/tensorflow:latest

LABEL maintainer="joaopedro@nindoo.ai"

RUN mkdir /root/app

WORKDIR /root/app/

RUN pip install --no-cache-dir -U pip

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY ./app .


CMD ["python", "main.py"]
