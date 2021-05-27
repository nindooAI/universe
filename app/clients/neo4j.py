from py2neo import Graph, Node, Relationship
from py2neo.matching import NodeMatcher
from py2neo.export import to_pandas_data_frame
from loguru import logger
import pandas as pd


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
        logger.info(dataframe.head())
        if "connections" in collection.keys():
            for connection in collection["connections"]:
                print(connection)
                for field in collection["connections"][connection]:
                    print(field)
                    if field in features:
                        features.remove(field)

        if len(features) > 0:
            features_dataframe = dataframe[features]
            features_dataframe[:] = 1

        cols = [col for col in features_dataframe.columns if '_id' not in col]
        clean_dataframe = features_dataframe[cols]
        clean_dataframe[:] = 1
        logger.info(clean_dataframe.head())

        return clean_dataframe

    def get_edges(self):
        pulling_query = """
                    MATCH (a)-->(b)
                    RETURN a.id AS source, b.id AS target
                    """
        edges = self.graph.run(pulling_query)
        return to_pandas_data_frame(edges)

    def set_emb(self, node_id, embedding):
        if type(node_id) == int:
            set_query = """
                    MATCH (a)
                    WHERE  ID(a) = $node_id
                    SET a.embedding = $embedding
                    """
        elif len(node_id) >= 1:
            set_query = """
                    UNWIND $node_id as id
                    UNWIND $embedding as emb
                    MATCH (a)
                    WHERE  ID(a) = id
                    SET a.embedding = emb
                    """
        resposta = self.graph.run(
            set_query, parameters={"node_id": node_id, "embedding": embedding})
        return resposta
