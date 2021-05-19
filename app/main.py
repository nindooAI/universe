from dotenv import load_dotenv
import requests
import json
import os
from loguru import logger
from clients.machine_learning import Universe
from scripts import pre_process
from clients.neo4j import n4j_client
from scripts.utils import make_dirs
from stellargraph.utils import plot_history
import uvicorn
from fastapi import FastAPI, Response, status, HTTPException
from stellargraph.utils import history, plot_history
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
        response = requests.get(os.getenv('CRAWLER_URL'), stream=True)
        response_txt = response.text
        articles = json.loads(response_txt)
        neo_client.populate_db(articles, source)

    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Erro ao atualizar o banco e treinar modelo")


@app.post('/retrain')
@logger.catch()
def retrain():
    """
    Retrain the model to update embeddings on neo4j.
    """
    logger.info('[!] Retreinando o modelo')
    logger.info('[1/x] Baixando dados do neo4j')
    try:
        users_dataframe = neo_client.get_nodes(
            label='User', features_dict={'miniBio': str})
        items_dataframe = neo_client.get_nodes(
            label='Blog', features_dict={'description': str})
        domains_dataframe = neo_client.get_nodes(
            label='AREA', features_dict={'name': str})
        interests_dataframe = neo_client.get_nodes(
            label='INTEREST', features_dict={'name': str})
        spaces_dataframe = neo_client.get_nodes(
            label='Space', features_dict={'name': str})
        sources_dataframe = neo_client.get_nodes(
            label='Source', features_dict={'name': str})
        shared_dataframe = neo_client.get_nodes(
            label='Shared', features_dict={})

        edges_dataframe = neo_client.get_edges()
        print(edges_dataframe.head())
        print(shared_dataframe.head())
    except Exception:
        raise HTTPException(
            status_code=404,
            detail="Erro ao puxar dados do neo4j")
    logger.info('[2/x] Pre-processando dados')
    nodes_dfs_list = [users_dataframe,
                      items_dataframe,
                      domains_dataframe,
                      interests_dataframe,
                      spaces_dataframe,
                      sources_dataframe,
                      shared_dataframe]
    merged_df = pre_process.merge_dfs(nodes_dfs_list)
    # head_nodes = list(nodes_dfs_dict.keys())
    # try:
    graph = pre_process.create_graph(edges_dataframe,
                                     nodes_df=merged_df)

    logger.info("Grafo: ")
    logger.info(graph.info())

    universe_client = Universe(graph=graph)
    logger.info("[3/x] Treinando modelo")
    history = universe_client.train()
    plot_history(history)

    # except Exception:
    #     raise HTTPException(
    #         status_code=404,
    #         detail="Erro ao preprocessar dados")

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


@app.post('/terraform')
def terraform():
    update_db()
    retrain()
    update_emb()


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=8000, host="0.0.0.0")
