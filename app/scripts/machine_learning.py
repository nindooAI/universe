#!/usr/bin/env python
# coding: utf-8

import numpy as np
from loguru import logger
from gensim.models import Word2Vec
from dotenv import load_dotenv
load_dotenv()


class Universe():
    def __init__(self, model_path=None):
        if model_path:
            try:
                self.model = self.load_model(model_path)
            except:
                print('Modelo não existe')

    @logger.catch
    def train(self, string_walks):
        logger.info('Treinando')
        weighted_model = Word2Vec(
            string_walks, vector_size=128, window=5, min_count=0, sg=1,
            workers=4, epochs=2
        )
        self.model = weighted_model
        return weighted_model

    @logger.catch
    def update_emb(self):
        logger.info('Atualizando embs')
        # Retrieve node embeddings and corresponding subjects
        node_ids = self.model.wv.index_to_key  # list of node IDs
        # the gensim ordering may not match the StellarGraph one, so rearrange
        int_ids = [int(node) for node in node_ids]

        embs = [list(map(float, list(self.model.wv[str(node)])))
                for node in int_ids]
        return embs, int_ids

    @ logger.catch
    def gen_emb(self, neighboors):
        array = neighboors[0][0]
        emb = np.mean(array, axis=0).tolist()
        return emb

    def save_model(self, model, model_path):
        model.save(model_path)

    def load_model(self, model_path):
        model = Word2Vec.load(model_path)
        return model
