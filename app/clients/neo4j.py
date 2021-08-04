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

    @logger.catch
    def get_nodes(self, label, data):
        for collection in data:
            collections = [collection["collection"] for collection in data]
            if label in collections and label not in list(collection["nodes"].keys()):
                features = collection["features"]
            else:
                features = collection["nodes"][label]["features"]

        nodes = self.node_matcher.match(label).all()
        dataframe = to_pandas_data_frame(nodes)
        nodes_ids = [node.identity for node in nodes]
        dataframe.index = nodes_ids

        if "connections" in collection.keys():
            for connections in collection["connections"]:
                for connection in connections.values():
                    for field in connection:
                        if field in features:
                            features.remove(field)
        if len(features) > 0:
            features_dataframe = dataframe[features]
            features_dataframe.index = nodes_ids
        else:
            features_dataframe = pd.DataFrame(index=nodes_ids)

        cols = [
            col for col in features_dataframe.columns
            if '_id' not in col or 'embedding' not in col]
        clean_dataframe = features_dataframe[cols]

        logger.info(' - '.join(['Dataframe clean', label]))
        logger.info(clean_dataframe.head())

        return clean_dataframe

    def get_edges(self, labels):
        pulling_query = """
                    MATCH (a)-->(b)
                    WHERE any(label in labels(a) WHERE label in $labels) \
                        and any(label in labels(b) WHERE label in $labels)
                    RETURN ID(a) AS source, ID(b) AS target
                    """
        edges = self.graph.run(pulling_query, parameters={"labels": labels})
        return to_pandas_data_frame(edges)

    @logger.catch()
    def set_emb(self, nodes_id: list, nodes_embeddings: list):
        logger.info("[*] Atualizando embeddings no banco de dados")
        batch_size = 1000
        count = 0
        transaction = self.graph.begin()
        for node_id, node_emb in zip(nodes_id, nodes_embeddings):
            count += 1
            node = self.node_matcher.get(node_id)
            try:
                label = list(node.labels)[0]
            except AttributeError:
                logger.error(node_id)
            set_query = """
                    MATCH (a:{node_label})
                    WHERE  ID(a) = $node_id
                    SET a.embedding = $node_emb
                    """.format(node_label=label)
            transaction.run(set_query, parameters={
                            "node_id": node_id,
                            "node_emb": node_emb})
            if count % batch_size == 0 or count == len(nodes_id):
                logger.info(
                    "Nós sendo atualizados {count}/{total}".format(count=count,
                                                                   total=len(nodes_id)))
                transaction.commit()
                transaction = self.graph.begin()

    @logger.catch()
    def get_neighboors(self, node_list, label_list):
        neighboors = [list(self.graph.run("""
                MATCH (a:{node_label})-[*1]-(b)
                WHERE  a.id = $nid
                return collect(ID(b)) as neighboors
                """.format(node_label=label), parameters={"nid": node_id}))
            for node_id, label in zip(node_list, label_list)]

        return neighboors

    @logger.catch()
    def get_node_features(self, node_list, label_list):
        logger.info('[*] Baixando features do nó')
        nodes = [self.node_matcher.match(label).where(id=node_id).first()
                 for node_id, label in zip(node_list, label_list)]

        return nodes

    @logger.catch()
    def get_recommendations(self, node_list, label_list, limit=15):
        results = [to_pandas_data_frame(self.graph.run("""
            MATCH (a:{node_label})-[*..2]-(b)
            WHERE a.id = $id AND EXISTS (a.embedding) AND EXISTS (b.embedding)
            RETURN DISTINCT b.id as id,labels(b) as label
            ORDER BY apoc.algo.cosineSimilarity(a.embedding,b.embedding)
            LIMIT $limit
        """.format(node_label=label),
            parameters={"id": node_id, "limit": limit}))
            for node_id, label in zip(node_list, label_list)]
        recommendations = []
        for result in results:
            node_rec = {}
            for index, row in result.iterrows():
                for label in row['label']:
                    try:
                        node_rec[label].append(row['id'])
                    except KeyError:
                        node_rec[label] = []
                        node_rec[label].append(row['id'])
            recommendations.append(node_rec)

        return recommendations
