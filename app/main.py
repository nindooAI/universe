from genericpath import exists
from logging import Logger, log
import matplotlib
from dotenv import load_dotenv
from matplotlib.pyplot import plot
import pandas as pd
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
from stellargraph.utils import plot_history
matplotlib.pyplot.switch_backend('Agg')
load_dotenv()


# Iniciando logs
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='./data/log_files/universe.log',
           backtrace=True, format=log_format, level='DEBUG')

data_path = './data/'
dev_dir = os.path.join(data_path, 'dev')
model_dir = os.path.join(data_path, 'model')
dash_dir = os.path.join(data_path, 'dash')

directories_list = [data_path, model_dir, dev_dir, dash_dir]

make_dirs(directories_list)


neo_client = n4j_client(connection_string=os.getenv(
    "N4J_URL"), user=os.getenv("N4J_USER"), password=os.getenv("N4J_PASS"))


app = FastAPI(title='Universe API', version='0.2',
              description='API com diversas funcoes do nindoo universe')


@ app.get('/')
@ app.get('/status')
def read_status():
    """
    Retorna status da API: ON ou OFF
    :return: Dicionario com chave 'message' e estado da api
    """
    logger.info('Usuario verificou estado da API')
    return {'message': 'Universe ON!'}


@ app.post('/retrain')
@ logger.catch()
def retrain(config: dict, from_api=True):
    """
    Retrain the model to update embeddings on neo4j.
    """
    model_config = config['model']
    model_path = os.path.join(
        model_config['model_path'], 'recomendation.model')

    data = config["data"]
    logger.info('[!] Retreinando o modelo')
    logger.info('[1/x] Baixando dados do neo4j')
    dataframes = [neo_client.get_nodes(collection)
                  for collection in data]

    logger.info('[2/x] Pre-processando dados')

    edges_dataframe = neo_client.get_edges(data)
    logger.info('[x] Datafrmae de ligações')
    logger.info(edges_dataframe.head)

    merged_df = pre_process.merge_dfs(dataframes)
    transformed_df = pre_process.transform(
        merged_df, config["preprocess"])
    logger.info('[x] Dataframe pré-processado')
    logger.info(transformed_df.columns)
    logger.info(transformed_df)

    logger.info('[*] Salvando grafos')
    transformed_df.to_csv(os.path.join(
        model_config['graph_path'], 'nodes.csv'))
    edges_dataframe.to_csv(os.path.join(
        model_config['graph_path'], 'edges.csv'))

    universe_client = Universe(edges_csv=os.path.join(
        model_config['graph_path'], 'edges.csv'),
        nodes_csv=os.path.join(
        model_config['graph_path'], 'nodes.csv'))

    logger.info("[3/x] Treinando modelo")
    history = universe_client.train()

    logger.info('[!] Salvando o modelo')
    universe_client.emb_model.save(model_path)

    loss_figure = plot_history(history, return_figure=True)
    loss_figure.savefig(os.path.join(dash_dir, 'loss.png'))

    emb_figure = plot_emb(universe_client)
    emb_figure.savefig(os.path.join(dash_dir, 'emb.png'))

    nodes_ids, nodes_embs = universe_client.update_emb()
    neo_client.set_emb(nodes_ids, nodes_embs)

    if not from_api:
        return universe_client
    else:
        return {'message': 'Retreino concluído'}

    # logger.info('[x] Treino finalizado e banco de dados atualizado.')


@ app.post('/update_emb')
@ logger.catch()
# gera emedding e deolve para o banco
def update_emb():
    logger.info('[*] Atualizando embeddings para neo4j')
    nodes_ids, nodes_embs = universe_client.update_emb()
    neo_client.set_emb(nodes_ids, nodes_embs)


@ logger.catch()
@ app.post('/get_emb')
def get_emb(node_list: list):
    logger.info('Gerando embedding para nós de IDs únicos:' + str(node_list))
    node_features = neo_client.get_node_features(node_list)
    ids = [node['id'] for node in node_features]
    neighboors = neo_client.get_neighboors(node_list)
    universe_client.update_graph(node_features, neighboors)
    embs = universe_client.gen_emb(ids)
    print(embs)
    neo_client.set_emb(node_list, embs)
    return {'message': "Embedding criado"}


def ml_init(config):
    try:
        model_config = config['model']
        model_path = os.path.join(
            model_config['model_path'], 'recomendation.model')
        if os.path.exists(model_path):
            universe_client = Universe(
                edges_csv=os.path.join(
                    model_config['graph_path'], 'edges.csv'),
                nodes_csv=os.path.join(
                    model_config['graph_path'], 'nodes.csv'),
                model_path=model_path)
            return universe_client
        else:
            universe_client = retrain(config, from_api=False)
    except Exception as error:
        logger.error('Erro ao iniciar o client de machine learning.')
        logger.error(error)


universe_client = ml_init(config=json.loads(os.getenv("CONFIG")))

if __name__ == "__main__":

    uvicorn.run(app, port=8000, host="0.0.0.0")
