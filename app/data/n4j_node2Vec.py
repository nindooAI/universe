#!/usr/bin/env python
# coding: utf-8

# # Pre-processando o banco de dados neo4j para alimentar algorítmo `node2vec` da stellargraph

# In[ ]:


from neo4j import GraphDatabase, basic_auth
import pandas as pd
import numpy as np
import os
import json
import collections
import sklearn
import stellargraph as sg
from sklearn.decomposition import PCA 
import os
import tensorflow as tf
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from loguru import logger
import time
from gensim.models import Word2Vec
driver = GraphDatabase.driver( os.getenv('N4J_URL'),  auth=basic_auth(os.getenv('N4J_USER'), os.getenv('N4J_PASS')))


# # Creating StellarGraph


# In[ ]:

@logger.catch
def pre_process():
    logger.info('Pre processando')
    pulling_query = """            MATCH (a)-->(b)
                RETURN ID(a) AS source, ID(b) AS target
                """
    with driver.session() as sess:
        result = sess.run(pulling_query)
        edges_db = pd.DataFrame([dict(row) for row in result])
    print(edges_db.shape)
    edges_db.head()

    graph = sg.StellarGraph(edges =  edges_db)
    print(graph.info())

### Random walks -> parte mais lenta
    start = time.time()
    logger.info('Random Walks')
    walk_length = 50  # maximum/ length of a random walk to use throughout this noteboo

    rw = sg.data.BiasedRandomWalk(graph)
    weighted_walks = rw.run(
        nodes=graph.nodes(),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=0.5,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=True,  # for weighted random walks
        seed=42,  # random seed fixed for reproducibility
    )
    #print("Number of random walks: {}".format(len(weighted_walks)))

    string_walks = [[str(n) for n in walk] for walk in weighted_walks]
    end = time.time()
    logger.info('Tempo das random Walks',end - start)
    return string_walks

### Training
def train(string_walks):
    logger.info('Treinando')
    weighted_model = Word2Vec(
        string_walks, size=128, window=5, min_count=0, sg=1, workers=1, iter=1
    )
    weighted_model.save("./app/data/word2vec.model")


### Embeddings

@logger.catch
def update_emb():
    logger.info('Atualizando embs')
    weighted_model = Word2Vec.load('./app/data/word2vec.model')
    # Retrieve node embeddings and corresponding subjects
    node_ids = weighted_model.wv.index2word  # list of node IDs
    weighted_node_embeddings = (
    weighted_model.wv.vectors
    )  # numpy.ndarray of size number of nodes times embeddings dimensionality
    # the gensim ordering may not match the StellarGraph one, so rearrange
    int_ids = [int(node) for node in node_ids]

    pulling_query = """            MATCH (a)
                WHERE  ID(a) = $id
                SET a.embedding = $emb
                """
    with driver.session() as sess:
        for node in int_ids:
            emb = map(float,list(weighted_model.wv[str(node)]))
            sess.run(pulling_query,{"id":node,"emb":emb})


@logger.catch
def gen_emb(nodeids):
    pulling_query = """        
                MATCH (a)-[*1]-(b)
                WHERE  ID(a) = $id
                return collect(b.embbedding)
                """
    with driver.session() as sess:
        array = sess.run(pulling_query,{"id":nodeids})
    
    mean = np.mean(array, axis=0)
    
    set_query = """        
                MATCH (a)
                WHERE  ID(a) = $id AND NOT a.embbedding
                SET a.embedding = $mean
                """
    with driver.session() as sess:
        array = sess.run(set_query,{"id":nodeids, "mean": mean})
