from py2neo import Graph, Node, Relationship
from py2neo.matching import NodeMatcher
from py2neo.export import to_pandas_data_frame
from loguru import logger
import pandas as pd
from itertools import islice


@logger.catch()
class n4j_client():
    def __init__(self, connection_string, user, password):
        self.graph = Graph(connection_string, auth=(user, password))
        self.node_matcher = NodeMatcher(self.graph)

    def to_data(self, result):
        return result.data()

    def get_nodes(self, collection):
        label = collection["collection"]
        features = collection["features"]
        nodes = self.node_matcher.match(label).all()
        dataframe = to_pandas_data_frame(nodes)
        dataframe.index = dataframe[collection["unique_id"]]
        if "connections" in collection.keys():
            for connection in collection["connections"]:
                for field in collection["connections"][connection]:
                    if field in features:
                        features.remove(field)

        if len(features) > 0:
            features_dataframe = dataframe[features]

        cols = [col for col in features_dataframe.columns if '_id' not in col]
        clean_dataframe = features_dataframe[cols]

        logger.info(' - '.join(['Dataframe raw', label]))
        logger.info(clean_dataframe.head())

        return clean_dataframe

    def get_edges(self):
        pulling_query = """
                    MATCH (a)-->(b)
                    RETURN a.id AS source, b.id AS target
                    """
        edges = self.graph.run(pulling_query)
        return to_pandas_data_frame(edges)

    def set_emb(self, nodes_id, nodes_embeddings):
        logger.info("Atualizando embeddings no banco de dados")
        if type(nodes_id) == int:
            set_query = """
                    MATCH (a)
                    WHERE  a.id = $node_id
                    SET a.embedding = $embedding
                    """
        else:
            set_query = """
                    UNWIND $node_id as id
                    UNWIND $embedding as emb
                    MATCH (a)
                    WHERE  a.id = id
                    SET a.embedding = emb
                    """
        batch_size = 100
        node_batch = []
        emb_batch = []
        count = 0
        transaction = self.graph.begin()
        for node_id, node_emb in zip(nodes_id, nodes_embeddings):
            count += 1
            node = self.node_matcher.match().where(id=node_id).first()
            node["embedding"] = node_emb
            transaction.merge(node)
            if count % batch_size == 0:
                self.graph.commit(transaction)
