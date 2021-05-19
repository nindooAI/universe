from py2neo import Graph, Node, Relationship
from py2neo.matching import NodeMatcher
from py2neo.export import to_pandas_data_frame
from loguru import logger
import pandas as pd
import sys


@logger.catch()
class n4j_client():
    def __init__(self, connection_string, user, password):
        self.graph = Graph(connection_string, auth=(user, password))
        self.node_matcher = NodeMatcher(self.graph)

    def to_data(self, result):
        return result.data()

    def get_nodes(self, label, features_dict):
        nodes = self.node_matcher.match(label).all()
        nodes_ids = [node.identity for node in nodes]
        dataframe = to_pandas_data_frame(nodes)
        dataframe.index = nodes_ids
        if len(features_dict) > 0:
            features_dataframe = dataframe[features_dict.keys()]
            features_dataframe[:] = 0
        else:
            return pd.DataFrame(index=nodes_ids)
        return features_dataframe

    def get_edges(self):
        pulling_query = """
                    MATCH (a)-->(b)
                    RETURN ID(a) AS source, ID(b) AS target
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

    def populate_db(self, response, source):
        tx = self.graph.begin()
        # Criando nó da fonte

        source_node = Node("Source", name=source)
        source_node.__primarykey__ = 'name'
        source_node.__primarylabel__ = "Source"
        tx.merge(source_node)
        tx.commit()

        # ### Super categorias
        # Criando nós de super categorias

        logger.info('[*] Super categorias')
        for element in response:
            if 'Label' in element.keys():
                self.graph.run("""MERGE (a:AREA {name: $label})
                    """, parameters={"label": element['Label']})

        # ### Interesses:
        # Crio os nós de Interesses,
        # se existir a label (crawler alterado), colcoar a label,
        # se não, apenas criar nó de interesse
        logger.info('[*] Interesses')
        for element in response:
            for interest in element['Category']:
                self.graph.run("""                MERGE (a:INTEREST {name: $name})
                    """, parameters={"name": interest})

        # ## Artigos
        # Primeiro crio os nós de artigos e depois conecto eles a
        # cada nó de categoria desses.
        # E adiciono as features de cada nó.

        logger.info('[*] Artigos')
        for element in response:
            try:
                self.graph.run("""MERGE (b:Data:Text:Blog {link: $link})
                    SET b.link = $link, b.image_url = $image_url, b.description = $description, b.date = $pub_date, b.title = $title
                    """, parameters={"link": element['Link'], "image_url": element['image'], "pub_date": element['PubDate'],
                                     "description": element['Description'], "title": element['Title']})
            except:
                pass

        # Limpando artigos repetidos

        self.graph.run("""
        MATCH (a:Blog),(b:Blog)
        WHERE toLower(a.link) <> tolower(b.link) AND id(a) <> id(b) AND tolower(a.title) = tolower(b.title)
        with collect(b) as nodes
        foreach (node in nodes | detach delete node)
        """
                       )

        self.graph.run("""
        MATCH (a:Blog),(b:Blog)
        WHERE toLower(a.title) = tolower(b.title) AND id(a) <> id(b)
        with collect(b) as nodes
        foreach (node in nodes | detach delete node)
        """
                       )

        # Criando ligações entre nós:
        logger.info('[*] Ligações')
        for element in response:
            for interest in element['Category']:
                if 'Label' in element.keys():
                    self.graph.run("""
                        MATCH (a:INTEREST {name: $name}),\
                        (b:Text {link: $link}),\
                        (c:AREA {name: $label}), (d:Source {name: $source})
                        MERGE (b)-[r:BELONGS_TO]->(a)
                        MERGE (b)-[:BELONGS_TO]->(c)
                        MERGE (a)<-[:INCLUDES]-(c)
                        MERGE (b)-[:IS_FROM]->(d)
                        """, parameters={"name": interest, "link": element['Link'],
                                         "label": element['Label'], "source": source})
                else:
                    self.graph.run("""                    MATCH (a:INTEREST {name: $name}),(b:Text {link: $link})
                        MERGE (b)-[r:BELONGS_TO]->(a)
                        """, parameters={"name": interest, "link": element['Link']})

        # ### Conectando áreas similares
        # SE mais de uma categoria aparece no artigo,
        # gerar conexões entre elas. Como tive que iterar
        # por todas, na outra célula apago os `self-loops`

        for element in response:
            for interest in element['Category']:
                if len(element['Category']) != 1:
                    self.graph.run("""                    MATCH (a:INTEREST {name: $name}), (b:INTEREST {name: $other}) 
                        WHERE NOT (a)-[:IS_RELATED]-(b) 
                        MERGE (a)-[:IS_RELATED]->(b)
                        """, parameters={"name": element['Category'][0], "other": interest})

        # Deletando ligações iguais
        for element in response:
            for interest in element['Category']:
                if len(element['Category']) >= 1:
                    self.graph.run("""
                        MATCH (a:INTEREST {name: $name})-[r:IS_RELATED]->(b:INTEREST\
                            {name: $other})
                        WHERE a.name = b.name
                        DELETE r
                        """,  parameters={"name": element['Category'][0], "other": interest})
