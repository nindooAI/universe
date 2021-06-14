from py2neo import Graph, Node, Relationship
from py2neo.matching import NodeMatcher
from py2neo.export import to_pandas_data_frame
from loguru import logger
import pandas as pd
from itertools import islice
from py2neo.bulk import merge_nodes


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
            for connections in collection["connections"]:
                for connection in connections.values():
                    for field in connection:
                        if field in features:
                            features.remove(field)
        if len(features) > 0:
            features_dataframe = dataframe[features]
        else:
            features_dataframe = pd.DataFrame(index=dataframe.index)

        cols = [
            col for col in features_dataframe.columns
            if '_id' not in col or 'embedding' not in col]
        clean_dataframe = features_dataframe[cols]

        logger.info(' - '.join(['Dataframe clean', label]))
        logger.info(clean_dataframe.head())

        return clean_dataframe

    def get_edges(self, data):
        labels = [entry["collection"] for entry in data]
        pulling_query = """
                    MATCH (a)-->(b)
                    WHERE any(label in labels(a) WHERE label in $labels) \
                        and any(label in labels(b) WHERE label in $labels)
                    RETURN a.id AS source, b.id AS target
                    """
        edges = self.graph.run(pulling_query, parameters={"labels": labels})
        return to_pandas_data_frame(edges)

    def set_emb(self, nodes_id: list, nodes_embeddings: list):
        logger.info("[*] Atualizando embeddings no banco de dados")
        batch_size = 1000
        count = 0
        transaction = self.graph.begin()
        for node_id, node_emb in zip(nodes_id, nodes_embeddings):
            count += 1
            node = self.node_matcher.match().where(id=node_id).first()
            label = list(node.labels)[0]
            set_query = """
                    MATCH (a:{label})
                    WHERE  a.id = $node_id
                    SET a.embedding = $node_emb
                    """.format(label=label)
            transaction.run(set_query, parameters={
                            "node_id": node_id,
                            "node_emb": node_emb})
            if count % batch_size == 0 or count == len(nodes_id):
                logger.info(
                    "Nós sendo atualizados {count}/{total}".format(count=count,
                                                                   total=len(nodes_id)))
                transaction.commit()
                transaction = self.graph.begin()

    def get_neighboors(self, node_list):
        neighboors = [list(self.graph.run("""
                MATCH (a)-[*1]-(b)
                WHERE  a.id = $nid
                return collect(b.id) as neighboors
                """, parameters={"nid": node_id}))
            for node_id in node_list]

        return neighboors

    def get_node_features(self, node_list):
        nodes = [self.node_matcher.match().where(id=node_id).first()
                 for node_id in node_list]

        return nodes
