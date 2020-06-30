import uvicorn
from fastapi import FastAPI
from data.create_db import populate_db
import data.n4j_node2Vec as n4j
from loguru import logger
import os
import pandas as pd

# Iniciando a instancia da API
app = FastAPI(title='Universe API', version='0.1',
                description='API com diversas funcoes do nindoo universe')

# Iniciando logs
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='app/data/log_files/logs.log',backtrace = True, format=log_format, level='DEBUG')

# Home da API
@app.get('/')

@app.get('/home')
def read_status():
    """
    Retorna status da API: ON ou OFF
    :return: Dicionario com chave 'message' e estado da api
    """
    logger.debug('Usuario verificou estado da API')
    return {'message': 'Universe ON!'}


@app.post('/update_db')
@logger.catch()
def update_db():
    """
    Puxa os dados do crawler e atualiza neo4j (talvez python não seja a melhor linguagem)
    """

    logger.info('Banco de dados sendo atualizado.')
    try:
        populate_db()
    except:
        return {"message", "Erro ao atualizar o banco de dados"}
    
    retrain()

@app.post('/retrain')
@logger.catch()
def retrain():
    """
    Retrain the model to update embeddings on neo4j.
    """
    logger.info('Retreinando o modelo')
    try:
        data = n4j.pre_process()
    except:
        return {"message", "Erro ao preprocessar dados"}   
    #logger.debug('Erro ao preprocessar dados')

    logger.info('Iniciando treino')
    try:
        n4j.train(data)
    except:
        return {"message", "Erro ao treinar modelo"}
        #logger.debug('Erro ao treinar')

    logger.info('Enviando embeddings para neo4j')
    try:
        n4j.update_emb()
    except:
        return {"message", "Erro ao tentar atualizar o neo4j"}
    #logger.debug('Erro em atualizar o DB')

    return {"message", "Modelo retreinado e banco de dados atualizado"}

@app.post('/emb')
@logger.catch()
# gera emedding e deolve para o banco
def get_emb(user_id):
    try:
        n4j.gen_emb(user_id)
        return { 'message': True}
    except:
        return Exception

if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=8000, host="0.0.0.0")
