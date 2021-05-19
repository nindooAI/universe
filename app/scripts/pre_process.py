import pandas as pd
from loguru import logger
import stellargraph as sg
import time
import tokenizers


@logger.catch()
# def tokenize(all_text):
def create_graph(edges_df, nodes_df):
    logger.info(['[*] Construindo grafo'])
    graph = sg.StellarGraph(nodes_df, edges=edges_df)
    return graph


def merge_dfs(nodes_dfs_list):
    return pd.concat(nodes_dfs_list)


def pre_process(nodes):

    # Random walks -> parte mais lenta
    start = time.time()
    logger.info('Random Walks')

    # walk_length = 5  # maximum/ length of a random walk to use
    # rw = sg.data.BiasedRandomWalk(graph)
    # weighted_walks = rw.run(
    #     nodes=graph.nodes(),  # root nodes
    #     length=walk_length,  # maximum length of a random walk
    #     n=10,  # number of random wxalks per root node
    #     # Defines (unormalised) probability, 1/p, of returning to source node
    #     p=0.5,
    #     # Defines (unormalised) probability, 1/q, for moving away from source
    #     q=0.5,
    #     weighted=True,  # for weighted random walks
    #     seed=42,  # random seed fixed for reproducibility
    # )

    # string_walks = [[str(n) for n in walk] for walk in weighted_walks]
    end = time.time()
    logger.info('Tempo das random Walks ' + str(end - start))
    return string_walks
