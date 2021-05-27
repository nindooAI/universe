import pandas as pd
from loguru import logger
import stellargraph as sg
import time


@logger.catch()
def create_graph(edges_df, nodes_df):
    logger.info(['[*] Construindo grafo'])
    graph = sg.StellarGraph(nodes_df, edges=edges_df)
    return graph


def merge_dfs(nodes_dfs_list):
    return pd.concat(nodes_dfs_list).fillna(0)


def transform(merged_df, preprocess_json):

    # Random walks -> parte mais lenta
    start = time.time()
    transformed = pd.DataFrame()
    logger.info('Pre-processando colunas do dataframe total')
    for value, key in preprocess_json["features"].items():
        if value == 'str':
            transformed[key] = string_transform(merged_df[key])
        if value == 'categorical':
            transformed[key] = categorical_transform(merged_df[key])

    end = time.time()
    logger.info('Tempo de pre-processamento ' + str(end - start))
    return transformed


def string_transform(series):
    pass


def categorical_transform(series):
    pass
