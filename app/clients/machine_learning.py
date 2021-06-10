#!/usr/bin/env python
# coding: utf-8

from fastcore.test import test
from networkx.classes.function import edges
from scripts import pre_process
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import os
import numpy as np
from loguru import logger
from pandas.core.frame import DataFrame
import stellargraph
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.mapper import (
    GraphSAGELinkGenerator,
    GraphSAGENodeGenerator,
)
import stellargraph as sg
from stellargraph.data import UnsupervisedSampler


load_dotenv()


class Universe():
    def __init__(self, edges_csv, nodes_csv, model_path=None, data_path=None,
                 n_walks=3, length=4, batch_size=1000):

        if model_path:
            logger.info(
                '[!] Carregando modelos e dados salvos da última versão.')
            self.emb_model = self.load_model(model_path)

        self.batch_size = batch_size
        self.edges_df = pd.read_csv(edges_csv)
        self.online_edges = self.edges_df.copy()

        self.transformed_df = pd.read_csv(nodes_csv, index_col='id')
        self.online_features = self.transformed_df.copy()
        self.graph = sg.StellarGraph(self.transformed_df, edges=self.edges_df)
        self.online_graph = self.graph

        self.sampler = UnsupervisedSampler(
            self.graph, nodes=list(self.graph.nodes()),
            length=length, number_of_walks=n_walks)

        self.base_generator = GraphSAGELinkGenerator(self.graph,
                                                     batch_size=self.batch_size,
                                                     num_samples=[5, 5])
        self.base_model = GraphSAGE(layer_sizes=[64, 64],
                                    generator=self.base_generator,
                                    bias=True, dropout=0.0, normalize="l2")

        self.emb_generator = GraphSAGENodeGenerator(self.graph,
                                                    batch_size=self.batch_size,
                                                    num_samples=[5, 5])

    def save_model(self, model, model_path):
        # model.save(model_path)
        pass

    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path)

    def save_data(self, model, data_path):
        pass

    @ logger.catch
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
    def gen_emb(self, ids):
        online_emb_generator = GraphSAGENodeGenerator(self.online_graph,
                                                      batch_size=self.batch_size,
                                                      num_samples=[5, 5])
        embs = self.emb_model.predict(online_emb_generator.flow(ids)).tolist()
        return embs

    def update_graph(self, node_features, neighboors):
        ids = [node['id'] for node in node_features]
        neighboors = [entry['neighboors'] for entry in neighboors[0]]
        logger.info(self.transformed_df.shape)
        new_features_df = pd.DataFrame([[-1 for i in self.transformed_df.columns]],
                                       index=ids, columns=self.transformed_df.columns)
        self.online_features = pd.concat(
            [self.transformed_df, new_features_df])
        logger.debug(self.transformed_df.shape)
        logger.debug(self.online_features.shape)

        new_edges = []
        for node_id, node_neighboors in zip(ids, neighboors):
            for neighboor in node_neighboors:
                new_edges.append([node_id, neighboor])

        new_edges_df = pd.DataFrame(
            new_edges, columns=self.edges_df.columns[1:])
        logger.debug(self.edges_df.shape)
        self.online_edges = pd.concat(
            [self.edges_df, new_edges_df], ignore_index=True)
        logger.debug(self.online_edges.shape)

        self.online_graph = sg.StellarGraph(
            self.online_features, edges=self.online_edges)
        logger.debug(self.online_graph.info())
