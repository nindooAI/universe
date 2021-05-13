from dotenv import load_dotenv
import requests
import json
import os
from loguru import logger
from scripts.machine_learning import Universe
from scripts.pre_process import pre_process
from clients.n4j_client import n4j_client
from scripts.utils import make_dirs
import uvicorn
from fastapi import FastAPI, HTTPException
load_dotenv()

# Iniciando logs
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='./data/log_files/universe.log',
           backtrace=True, format=log_format, level='DEBUG')

data_path = './data/'
model_dir = './data/models/'
model_path = './data/models/node2vec.model'
directories_list = [data_path, model_dir]

make_dirs(directories_list)

# Iniciando a instancia da API
app = FastAPI(title='Universe API', version='0.1',
              description='API com diversas funcoes do nindoo universe')


neo_client = n4j_client(connection_string=os.getenv(
    "N4J_URL"), user=os.getenv("N4J_USER"), password=os.getenv("N4J_PASS"))

universe_client = Universe(model_path=model_path)


@app.get('/')
@app.get('/status')
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
    Puxa os dados do crawler e atualiza neo4j
    (talvez python não seja a melhor linguagem)
    """
    source = 'Medium'
    logger.info("Atualizando grafo do: " + source)

    try:
        logger.info('[*] Puxando dados do crawler')
        response = requests.get(os.getenv('CRAWLER_URL'))
        response_txt = response.text
        articles = json.loads(response_txt)
        neo_client.populate_db(articles, source)

    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Erro ao atualizar o banco e treinar modelo")

    retrain()


@app.post('/retrain')
@logger.catch()
def retrain():
    """
    Retrain the model to update embeddings on neo4j.
    """
    logger.info('Retreinando o modelo')
    logger.info('[*] Baixando dados do neo4j')
    try:
        nodes = neo_client.get_all_nodes()
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Erro ao puxar dados do neo4j")
    try:
        string_walks = pre_process(nodes)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Erro ao preprocessar dados")

    logger.info('Iniciando treino')
    try:
        new_model = universe_client.train(string_walks)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Erro ao preprocessar dados")

    logger.info("[*] Salvando o modelo")
    try:
        universe_client.save_model(new_model, model_path)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Erro ao salvar modelo")

    update_emb()


@app.post('/update_emb')
@logger.catch()
# gera emedding e deolve para o banco
def update_emb():
    logger.info('[*] Atualizando embeddings para neo4j')
    try:
        embs, int_ids = universe_client.update_emb()
        queries = neo_client.set_emb(int_ids, embs)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Erro ao tentar atualizar os embeddings no neo4j")

    return {"message", "Embeddings atualizados"}


@app.post('/get_emb')
@logger.catch()
def get_emb(node_id):
    logger.info('Gerando embedding para nó de ID' + str(node_id))
    try:
        neighboors = list(neo_client.worker.run("""
                MATCH (a)-[*1]-(b)
                WHERE  ID(a) = $nid
                return collect(b.embedding)
                """, {"nid": int(node_id)}))
        emb = universe_client.gen_emb(neighboors)
        neo_client.set_emb(list(int(node_id)), emb)
        return {'message': "Embedding criado"}
    except Exception:
        return HTTPException(
            status_code=404,
            detail="Erro ao preprocessar dados")


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=8000, host="0.0.0.0")
