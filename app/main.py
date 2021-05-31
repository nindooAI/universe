import matplotlib
from dotenv import load_dotenv
from matplotlib.pyplot import plot
import requests
import json
import os
from loguru import logger
from clients.machine_learning import Universe
from scripts import pre_process
from clients.neo4j import n4j_client
from scripts.utils import make_dirs
from scripts.interpret import plot_emb
from stellargraph.utils import plot_history
import uvicorn
from fastapi import FastAPI, Response, status, HTTPException
from stellargraph.utils import history, plot_history
load_dotenv()
# matplotlib.pyplot.switch_backend('Agg')

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


@app.get('/')
@app.get('/status')
def read_status():
    """
    Retorna status da API: ON ou OFF
    :return: Dicionario com chave 'message' e estado da api
    """
    logger.debug('Usuario verificou estado da API')
    return {'message': 'Universe ON!'}


@app.post('/retrain')
@logger.catch()
def retrain(db_json: dict):
    """
    Retrain the model to update embeddings on neo4j.
    """
    logger.info('[!] Retreinando o modelo')
    logger.info('[1/x] Baixando dados do neo4j')
    dataframes = [neo_client.get_nodes(collection)
                  for collection in db_json["data"]]

    edges_dataframe = neo_client.get_edges()

    logger.info('[2/x] Pre-processando dados')
    preprocess_json = {"features": {"name": 'string',
                                    "type": 'categorical',
                                    'description': 'string',
                                    'resource_type': 'categorial',
                                    'slug': 'categorical'}}
    merged_df = pre_process.merge_dfs(dataframes)
    logger.info('merged_df')
    logger.info(merged_df.head())

    transformed_df = pre_process.transform(merged_df, preprocess_json)
    logger.info('transformed_df')
    logger.info(transformed_df.columns)
    logger.info(transformed_df)

    transformed_df.to_csv('debug.csv')

    graph = pre_process.create_graph(edges_dataframe,
                                     nodes_df=transformed_df)
    logger.info("Grafo: ")
    logger.info(graph.info())

    universe_client = Universe(graph=graph)
    logger.info("[3/x] Treinando modelo")
    history = universe_client.train()
    figure = plot_history(history, return_figure=True)
    # figure.savefig('loss.png')
    # plot_emb(universe_client)
    nodes_ids, nodes_embs = universe_client.update_emb()
    neo_client.set_emb(nodes_ids, nodes_embs)


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


@logger.catch()
@app.post('/get_emb')
def get_emb(node_id: int, response: Response):
    logger.info('Gerando embedding para nó de ID' + str(node_id))
    # try:
    neighboors = list(neo_client.graph.run("""
            MATCH (a)-[*1]-(b)
            WHERE  ID(a) = $nid
            return collect(b.embedding)
            """, parameters={"nid": int(node_id)}))
    emb = universe_client.gen_emb(neighboors)
    neo_client.set_emb(node_id, emb)
    return {'message': "Embedding criado"}
    # except Exception as ex:
    #     response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    #     return HTTPException(
    #         status_code=response.status_code,
    #         detail=ex)


# @app.post('/terraform')
# def terraform(db_json: dict):
#     retrain(db_json)
#     update_emb()


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping

    # uvicorn.run(app, port=8000, host="0.0.0.0")
    retrain({"data": [
        {"collection": "lessons", "unique_id": "id", "features": [
            "type", "name", "theme_id"], "connections":{"themes": ["lesson-themes", "theme_id"]}},
        {"collection": "data", "unique_id": "id", "features": [
                                            "resource_type", "resource_id", "lesson_id"], "connections":{"lessons": ["data-lesson", "lesson_id"]}},
        {"collection": "courses", "unique_id": "id", "features": [
            "name", "description", "slug"]},
        {"collection": "themes", "unique_id": "id", "features": [
            "name", "course_id", "type"], "connections":{"courses": ["themes-course", "course_id"]}}
    ]})
