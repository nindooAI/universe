#!/usr/bin/env python
# coding: utf-8

from networkx.classes import graph
from networkx.classes.graph import Graph
import pandas as pd
import numpy as np
from loguru import logger
from pandas.core.frame import DataFrame
from stellargraph import data
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.mapper import (
    GraphSAGELinkGenerator,
    GraphSAGENodeGenerator,
)
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from matplotlib import pyplot as plt
from dotenv import load_dotenv
load_dotenv()


class Universe():
    def __init__(self, graph, model_path=None, data_path=None,
                 n_walks=1, length=2, batch_size=500):

        if data_path:
            self.graph = self.load_data(data_path)
        else:
            self.graph = graph

        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.model = None

        self.sampler = UnsupervisedSampler(
            self.graph, nodes=list(self.graph.nodes()),
            length=length, number_of_walks=n_walks)

        self.base_generator = GraphSAGELinkGenerator(self.graph,
                                                     batch_size=batch_size,
                                                     num_samples=[5, 5])
        self.base_model = GraphSAGE(layer_sizes=[64, 64],
                                    generator=self.base_generator,
                                    bias=True, dropout=0.0, normalize="l2")

        self.emb_generator = GraphSAGENodeGenerator(self.graph,
                                                    batch_size=batch_size,
                                                    num_samples=[5, 5])

    def save_model(self, model, model_path):
        # model.save(model_path)
        pass

    def load_model(self, model_path):
        # return model
        pass

    def save_data(self, model, data_path):
        pass

    def load_data(self, data_path):
        # return stellar_graph
        pass

    @logger.catch
    def train(self, epochs=2):

        x_in, x_out = self.base_model.in_out_tensors()

        train_gen = self.base_generator.flow(self.sampler)

        prediction = link_classification(output_dim=1, output_act="sigmoid",
                                         edge_embedding_method="ip")(x_out)

        model = tf.keras.Model(inputs=x_in, outputs=prediction)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=["binary_accuracy"],
        )
        es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
        history = model.fit(
            train_gen,
            epochs=epochs,
            verbose=1,
            use_multiprocessing=True,
            workers=4,
            callbacks=[es])

        x_in_src = x_in[0::2]
        x_out_src = x_out[0]
        self.emb_model = tf.keras.Model(inputs=x_in_src, outputs=x_out_src)

        return history

    @ logger.catch
    def update_emb(self):
        logger.info("[*] Gerando Embeddings para todos os nós")
        ids = list(self.graph.nodes())
        emb = self.emb_model.predict(self.emb_generator.flow(ids)).tolist()
        return ids, emb

    @ logger.catch
    def gen_emb(self, neighboors):
        array = neighboors[0][0]
        emb = np.mean(array, axis=0).tolist()
        return emb
