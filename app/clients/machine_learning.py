#!/usr/bin/env python
# coding: utf-8

from networkx.classes.graph import Graph
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.manifold import TSNE
import stellargraph as sg
from stellargraph.mapper import (
    CorruptedGenerator,
    GraphSAGENodeGenerator,
    FullBatchNodeGenerator
)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from stellargraph import StellarGraph
from stellargraph.layer import DeepGraphInfomax, GraphSAGE
from matplotlib import pyplot as plt
from dotenv import load_dotenv
load_dotenv()


class Universe():
    def __init__(self, graph, model_path=None):
        self.graph = graph
        self.generator = GraphSAGENodeGenerator(self.graph,
                                                batch_size=1000,
                                                num_samples=[5])
        self.base_model = GraphSAGE(layer_sizes=[128], activations=[
            "relu"], generator=self.generator)

    @logger.catch
    def train(self, epochs=5, reorder=lambda sequence,
              subjects: subjects):
        corrupted_generator = CorruptedGenerator(self.generator)
        gen = corrupted_generator.flow(self.graph.nodes())
        infomax = DeepGraphInfomax(self.base_model, corrupted_generator)

        x_in, x_out = infomax.in_out_tensors()

        model = Model(inputs=x_in, outputs=x_out)
        model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits,
                      optimizer=Adam(lr=1e-3))
        es = EarlyStopping(monitor="loss", min_delta=0, patience=20)
        history = model.fit(gen, epochs=epochs, verbose=0, callbacks=[es])
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
        return history

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
