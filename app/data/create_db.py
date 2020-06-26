#!/usr/bin/env python
# coding: utf-8

# # Discover DB
# 

# ## Imports e configurações iniciais

# In[ ]:


import pandas as pd
import os
import requests
import json
from neo4j import GraphDatabase, basic_auth
import wikipedia
from loguru import logger

driver = GraphDatabase.driver(os.getenv('N4J_URL'),  auth=basic_auth(os.getenv('N4J_USER'),os.getenv('N4J_PASS')))
# Pega dados do crawler

# In[ ]:
@logger.catch
def populate_db():
    print('Crawler')
    response_txt = requests.get(os.getenv('CRAWLER_URL')).text 
    response = json.loads(response_txt)


    # ## Descobrimento de dados
    # Descomentar para entender a estrutura

    # In[ ]:


    # for element in response:
    #     print(element.keys())
    #     print(element['Category'])
    #     print(element['Description'])
    #     print(element['Link'])
    #     print(element['PubDate'])
    #     print(element['Title'])
    #     print(element['image'])
    #     break


    # ## Conectando ao db

    # In[ ]:


    


    # ### Super categorias
    # Criando nós de super categorias

    # In[ ]:


    print('Super categorias')
    with driver.session() as sess:
        for element in response:
            if 'Label' in element.keys():
                sess.run("""                MERGE (a:AREA {name: $label})
                    """, {"label":element['Label']})


    # ### Interesses:
    # Crio os nós de Interesses, se existir a label (crawler alterado), colcoar a label, se não, apenas criar nó de interesse
    # 
    # #### atualizar depois de trnasicionar db

    # In[ ]:


    print('Interesses')
    with driver.session() as sess:
        for element in response:
            for interest in element['Category']:
                sess.run("""                MERGE (a:INTEREST {name: $name})
                    """, {"name":interest})


    # ## Artigos 
    # Primeiro crio os nós de artigos e depois conecto eles a cada nó de categoria desses. E adiciono as features de cada nó.
    # 

    # In[ ]:


    print('Artigos e interesses')
    with driver.session() as sess:
        for element in response:
            sess.run("""            MERGE (b:Data:Text:Blog {link: $link})
                ON CREATE SET b.link = $link, b.image_url = $image_url, b.description = $description, b.date = $pub_date
                """, {"link":element['Link'],"image_url":element['image'],"pub_date":element['PubDate'],
                    "description":element['Description'], "title":element['Title']})


    # ### Criando ligações entre nós:

    # In[ ]:


    with driver.session() as sess:
        for element in response:
            for interest in element['Category']:
                if 'Label' in element.keys():
                    sess.run("""                    MATCH (a:INTEREST {name: $name}),(b:Text {link: $link}),(c:AREA {name: $label})
                        MERGE (b)-[r:BELONGS_TO]->(a)
                        MERGE (b)-[:BELONGS_TO]->(c)
                        MERGE (a)<-[:INCLUDES]-(c)
                        """, {"name":interest, "link":element['Link'], "label":element['Label']})
                else:
                    sess.run("""                    MATCH (a:INTEREST {name: $name}),(b:Text {link: $link})
                        MERGE (b)-[r:BELONGS_TO]->(a)
                        """, {"name":interest, "link":element['Link']})


    # ### Conectando áreas similares
    # SE mais de uma categoria aparece no artigo, gerar conexões entre elas. Como tive que iterar por todas, na outra célula apago os `self-loops`

    # In[ ]:


    with driver.session() as sess:
        for element in response:
            for interest in element['Category']:
                if len(element['Category']) != 1:    
                    sess.run("""                    MATCH (a:INTEREST {name: $name}), (b:INTEREST {name: $other}) 
                        WHERE NOT (a)-[:IS_RELATED]-(b) 
                        MERGE (a)-[:IS_RELATED]->(b)
                        """, {"name":element['Category'][0], "other":interest})


    # In[ ]:


    ### Deletando ligações iguais
    with driver.session() as sess:
        for element in response:
            for interest in element['Category']:
                if len(element['Category']) >=1:    
                    sess.run("""                    MATCH (a:INTEREST {name: $name})-[r:IS_RELATED]->(b:INTEREST {name: $other}) 
                        WHERE a.name = b.name
                        DELETE r
                        """, {"name":element['Category'][0], "other":interest})


    # ### Wikipedia
    # Uso a api da wikipedia para gerar descrições nos nós de categorias, salvo o arquivo 'wiki.json' para não precisar fazer request denovo, eles são a parte mais demoradade desse código'

    # In[ ]:


    print('Wikipedia')
    with open('./app/data/wiki.json','r') as fp:
        pre_dict = json.load(fp)


    # In[ ]:





    # In[ ]:


    lista = []
    for element in response:
            for interest in element['Category']:
                if interest not in pre_dict.keys():
                    lista.append(interest)
    interests = set(lista)
    descriptions = pre_dict
    for element in interests:
        if element not in descriptions.keys():
            try:
                pagina = wikipedia.page(element)
                descriptions[element] = pagina.summary
            except:
                descriptions[element] = 'Not Found'
            


    # In[ ]:


    with open('./app/data/wiki.json', 'w') as fp:
        json.dump(descriptions, fp)


    # In[ ]:


    with driver.session() as sess:
        for element in descriptions.keys():
            sess.run("""                MATCH (b:INTEREST {name: $name})
                    SET b.description = $description
                    """, {"name":element,"description":descriptions[element]})
        


    # In[ ]:



    # for key,value in descriptions.items():
    #     if value == 'Not Found':
    #         print(key)


# In[ ]:
if __name__ == '__main__':
    update_db()


