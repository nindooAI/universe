from logging import exception
from py2neo import Graph
from loguru import logger


@logger.catch()
class n4j_client():
    def __init__(self, connection_string, user, password):
        self.worker = Graph(connection_string, auth=(user, password))

    def populate_db(self, response, source):

        # Criando nó da fonte

        self.worker.run("""
                MERGE (a:Source {name:$source})
                        """, {"source": source})

        # ### Super categorias
        # Criando nós de super categorias

        logger.info('[*] Super categorias')
        for element in response:
            if 'Label' in element.keys():
                self.worker.run("""MERGE (a:AREA {name: $label})
                    """, {"label": element['Label']})

        # ### Interesses:
        # Crio os nós de Interesses,
        # se existir a label (crawler alterado), colcoar a label,
        # se não, apenas criar nó de interesse
        logger.info('[*] Interesses')
        for element in response:
            for interest in element['Category']:
                self.worker.run("""                MERGE (a:INTEREST {name: $name})
                    """, {"name": interest})

        # ## Artigos
        # Primeiro crio os nós de artigos e depois conecto eles a
        # cada nó de categoria desses.
        # E adiciono as features de cada nó.

        logger.info('[*] Artigos')
        for element in response:
            try:
                self.worker.run("""MERGE (b:Data:Text:Blog {link: $link})
                    SET b.link = $link, b.image_url = $image_url, b.description = $description, b.date = $pub_date, b.title = $title
                    """, {"link": element['Link'], "image_url": element['image'], "pub_date": element['PubDate'],
                          "description": element['Description'], "title": element['Title']})
            except:
                pass

        # Limpando artigos repetidos

        self.worker.run("""
        MATCH (a:Blog),(b:Blog)
        WHERE toLower(a.link) <> tolower(b.link) AND id(a) <> id(b) AND tolower(a.title) = tolower(b.title)
        with collect(b) as nodes
        foreach (node in nodes | detach delete node)
        """
                        )

        self.worker.run("""
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
                    self.worker.run("""                    
                        MATCH (a:INTEREST {name: $name}),(b:Text {link: $link}),(c:AREA {name: $label}), (d:Source {name: $source})
                        MERGE (b)-[r:BELONGS_TO]->(a)
                        MERGE (b)-[:BELONGS_TO]->(c)
                        MERGE (a)<-[:INCLUDES]-(c)
                        MERGE (b)-[:IS_FROM]->(d)
                        """, {"name": interest, "link": element['Link'], "label": element['Label'], "source": source})
                else:
                    self.worker.run("""                    MATCH (a:INTEREST {name: $name}),(b:Text {link: $link})
                        MERGE (b)-[r:BELONGS_TO]->(a)
                        """, {"name": interest, "link": element['Link']})

        # ### Conectando áreas similares
        # SE mais de uma categoria aparece no artigo, gerar conexões entre elas. Como tive que iterar por todas, na outra célula apago os `self-loops`

        for element in response:
            for interest in element['Category']:
                if len(element['Category']) != 1:
                    self.worker.run("""                    MATCH (a:INTEREST {name: $name}), (b:INTEREST {name: $other}) 
                        WHERE NOT (a)-[:IS_RELATED]-(b) 
                        MERGE (a)-[:IS_RELATED]->(b)
                        """, {"name": element['Category'][0], "other": interest})

        # Deletando ligações iguais
        for element in response:
            for interest in element['Category']:
                if len(element['Category']) >= 1:
                    self.worker.run("""                    MATCH (a:INTEREST {name: $name})-[r:IS_RELATED]->(b:INTEREST {name: $other}) 
                        WHERE a.name = b.name
                        DELETE r
                        """, {"name": element['Category'][0], "other": interest})

    def get_all_nodes(self):
        pulling_query = """
                    MATCH (a)-->(b)
                    RETURN ID(a) AS source, ID(b) AS target
                    """
        result = self.worker.run(pulling_query)
        return result

    def set_emb(self, node_id, embedding):
        if len(node_id) == 1:
            set_query = """
                    MATCH (a)
                    WHERE  ID(a) = $id
                    SET a.embedding = $emb
                    """
        else:
            set_query = """
                    UNWIND $node_id as id
                    UNWIND $embedding as emb
                    MATCH (a)
                    WHERE  ID(a) = id
                    SET a.embedding = emb
                    """
        resposta = self.worker.run(set_query, {"node_id": node_id, "embedding": embedding})
        return resposta
