from typing import Counter
import pandas as pd
from loguru import logger
import stellargraph as sg
import time
from fastai.text.data import tokenize_df, make_vocab, Numericalize, pad_chunk
from fastai.torch_core import to_np


@logger.catch()
def create_graph(edges_df, nodes_df):
    logger.info(['[*] Construindo grafo'])
    graph = sg.StellarGraph(nodes_df, edges=edges_df)
    return graph


def create_pseudo_labels(merged_df, edges_df):
    pass


def merge_dfs(nodes_dfs_list):
    return pd.concat(nodes_dfs_list).fillna(-1)


def transform(merged_df, preprocess_json):
    start = time.time()
    transformed = merged_df.copy()
    logger.info('Pre-processando colunas do dataframe total')
    string_cols = []
    categorical_cols = []
    for key, value in preprocess_json["features"].items():
        if value == 'string':
            feature_columns, tokenized, counter = string_transform(
                merged_df, key)
            transformed[feature_columns] = tokenized[feature_columns]
            transformed[key] = tokenized['tok_' + key]
        if value == 'categorical':
            codes, uniques = categorical_transform(
                merged_df, key)
            transformed['cat_'+key] = codes
        transformed = transformed.drop(columns=key)

    end = time.time()
    logger.info('Tempo de pre-processamento ' + str(end - start))
    return transformed


def string_transform(dataframe, string_col):
    tok_text_col = 'tok_' + string_col
    tok_df, counter = tokenize_df(dataframe, string_col, n_workers=2,

                                  tok_text_col=tok_text_col)

    vocab = make_vocab(counter, min_freq=3, max_vocab=60000)

    num = Numericalize(vocab)

    tok_df[tok_text_col] = tok_df[tok_text_col].apply(num)
    max_len = max(tok_df[tok_text_col].apply(len))
    feature_columns = [tok_text_col+str(i) for i in range(max_len)]

    tok_df[tok_text_col] = tok_df[tok_text_col].map(
        lambda x: pad_chunk(x, pad_idx=-1, pad_len=max_len, pad_first=False))
    tok_df[tok_text_col] = tok_df[tok_text_col].apply(to_np)

    tok_df[feature_columns] = pd.DataFrame(
        tok_df[tok_text_col].tolist(), index=tok_df.index)

    return feature_columns, tok_df, vocab


def categorical_transform(dataframe, category_col):
    codes, uniques = pd.factorize(dataframe[category_col], sort=False)
    return codes, uniques
