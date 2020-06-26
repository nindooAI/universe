FROM tensorflow/tensorflow:latest-py3

LABEL maintainer="joaopedro@nindoo.ai"

RUN mkdir -p /home/universe-api/

WORKDIR /home/universe-api/

RUN pip install --no-cache-dir -U pip

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY . .

ENV CRAWLER_URL='https://spxa6xmc58.execute-api.us-west-2.amazonaws.com/prod/'
ENV N4J_URL='bolt://104.248.235.192:7687/'
ENV N4J_USER='neo4j'
ENV N4J_PASS='nindoo123'

CMD ["python", "app/main.py"]
