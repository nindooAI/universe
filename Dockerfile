FROM python:3.7-slim

LABEL maintainer="joaopedro@nindoo.ai"

RUN mkdir /root/app

WORKDIR /root/app/


RUN pip install --no-cache-dir -U pip

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY ./app .


CMD ["python", "main.py"]
