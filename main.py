import uvicorn
from fastpai import FastAPI
import data.create_db 
import data.n4j_node2Vec 
from loguro import logger

# Iniciando a instancia da API
app = FastAPI(title='Universe API', version='0.1,
                description='API com diversas funcoes do nindoo universe')

# Iniciando logs
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='app/data/log_files/logs.log', format=log_format, level='DEBUG', compression='zip')

# Home da API
@app.get('/')

@app.get('/home')
def read_status():
    """
    Retorna status da API: ON ou OFF
    :return: Dicionario com chave 'message' e estado da api
    """
    logger.debug('Usuario verificou estado da API')
    return {'message': 'Universe de ON!'}


@app.post('/update_db')
@logger.catch()
def update_db():
    """
    Puxa os dados do crawler e atualiza neo4j (talvez python não seja a melhor linguagem)
    """

    logger.info('Banco de dados sendo atualizado.')
    update_db.run()
    logger.debug('Erro ao enviar embeddings para neo4j')
    retrain()
    
@app.post('/retrain')
def retrain():
    """
    Retrain the model to update embeddings on neo4j.
    """
    logger.info('Retreinando o modelo')
    data = n4j_node2Vec.pre_process()
    logger.debug('Erro ao preprocessar dados')

    logger.info('Iniciando treino')
    n4j_node2Vec.train(data)
    logger.debug('Erro ao treinar')

    logger.info('Enviando embeddings para neo4j')
    n4j_node2Vec.update_emb()
    logger.debug('Erro em atualizar o DB')

@app.post('/emb')
@logger.catch()
# gera emedding e deolve para o banco
def get_emb(user_id):
    n4j_node2Vec.get_emb(user_id)
    return { 'message': True}


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=8000, host="0.0.0.0")