#!/usr/bin/env python
# coding: utf-8

from networkx.classes.graph import Graph
import pandas as pd
import numpy as np
from loguru import logger
from pandas.core.frame import DataFrame
from stellargraph.layer import GraphSAGE, link_classification
from stellargraph.mapper import (
    GraphSAGELinkGenerator,
    FullBatchNodeGenerator
)
from stellargraph.data import UniformRandomWalk
from stellargraph.data import UnsupervisedSampler

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CosineSimilarity
import tensorflow as tf
from matplotlib import pyplot as plt
from dotenv import load_dotenv
load_dotenv()


class Universe():
    def __init__(self, graph, n_walks=1, length=3, model_path=None, data_path=None):
        self.graph = graph

        self.sampler = UnsupervisedSampler(
            self.graph, nodes=list(self.graph.nodes()),
            length=length, number_of_walks=n_walks)

        self.generator = GraphSAGELinkGenerator(self.graph,
                                                batch_size=500,
                                                num_samples=[5, 5])
        self.base_model = GraphSAGE(layer_sizes=[32, 32],
                                    generator=self.generator,
                                    bias=True, dropout=0.0, normalize="l2")
        # if data_path:
        #     DataFrame = pd.

    @logger.catch
    def train(self, epochs=1):

        x_in, x_out = self.base_model.in_out_tensors()

        train_gen = self.generator.flow(self.sampler)

        prediction = link_classification(output_dim=1, output_act="sigmoid",
                                         edge_embedding_method="ip")(x_out)

        model = tf.keras.Model(inputs=x_in, outputs=prediction)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-3),
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
        # x_emb_in, x_emb_out = self.base_model.in_out_tensors()
        # emb_model = Model(inputs=x_emb_in, outputs=x_emb_out)
        # x_emb_in, x_emb_out = self.base_model.in_out_tensors()
        # # for full batch models, squeeze out the batch dim (which is 1)
        # if self.generator.num_batch_dims() == 2:
        #     x_emb_out = tf.squeeze(x_emb_out, axis=0)
        # fullbatch_generator = FullBatchNodeGenerator(self.graph, sparse=False)
        # embeddings = emb_model.predict(
        #     fullbatch_generator.flow(self.graph.nodes()))
        # trans = TSNE(n_components=2)
        # emb_transformed = pd.DataFrame(
        #     trans.fit_transform(embeddings), index=self.graph.nodes())
        # # so we need to get everything lined up correctly
        # ordered_test_subjects = reorder(test_gen, test_subjects)
        # ordered_train_subjects = reorder(train_gen, train_subjects)

        # lr = LogisticRegression(multi_class="auto", solver="lbfgs")
        # lr.fit(train_embeddings, ordered_train_subjects)

        # y_pred = lr.predict(test_embeddings)
        # acc = (y_pred == ordered_test_subjects).mean()
        return model, history

    @ logger.catch
    def update_emb(self):
        logger.info('Atualizando embs')
        # Retrieve node embeddings and corresponding subjects
        node_ids = self.base_model.wv.index_to_key  # list of node IDs
        # the gensim ordering may not match the StellarGraph one, so rearrange
        int_ids = [int(node) for node in node_ids]

        embs = [list(map(float, list(self.base_model.wv[str(node)])))
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
