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
import config


# ## Data loading
# 
# Pega dados do crawler

# In[ ]:


print('Crawler')
def get_crawler(url):
    response = requests.get(url)
    return response.text

response = json.loads(get_crawler(os.getenv('CRAWLER_URL')))


# ## Descobrimento de dados
# Descomentar para entender a estrutura

# In[ ]:


# soma = 0
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


driver = GraphDatabase.driver(os.getenv('N4J_URL'),  auth=basic_auth(os.getenv('N4J_USER'),os.getenv('N4J_PASS')))
sess = driver.session()


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
            if 'Label' in element.keys():
                sess.run("""                    MATCH (b:AREA {name:$label})
                    MERGE (a:INTEREST {name: $name})<-[r:INCLUDES]-(b)
                    """, {"name":interest,"label":element['Label']})
            else:
                sess.run("""                    MERGE (a:INTEREST {name: $name})
                    """, {"name":interest})


# ## Artigos ligados à Interesses e areas
# Primeiro crio os nós de artigos e depois conecto eles a cada nó de categoria desses. E adiciono as features de cada nó.
# 

# In[ ]:


print('Artigos e interesses')
with driver.session() as sess:
    for element in response:
        sess.run("""            MERGE (b:Data:Text:Blog {title: $title})
            SET b.link = $link, b.image_url = $image_url, b.description = $description, b.date = $pub_date
            """, {"link":element['Link'],"image_url":element['image'],"pub_date":element['PubDate'],
                  "description":element['Description'], "title":element['Title']})


# In[ ]:


with driver.session() as sess:
    for element in response:
        for interest in element['Category']:
            sess.run("""                
		MATCH (a:INTEREST {name: $name}),(b:Text {title: $title}),(c:AREA {name: $label})
                MERGE (b)-[r:BELONGS_TO]->(a)
                MERGE (b)-[:BELONGS_TO]->(c)
                """, {"name":interest, "title":element['Title'], "label":element['Label']})


# In[ ]:


#     with driver.session() as sess:
#         for element in response:
#             sess.run("""\
#                     MATCH (b:Text {title: $title})
#                     SET b.link = $link, b.image_url = $image_url, b.description = $description, b.date = $pub_date
#                     SET b:Text:Blog
#                     """, {"link":element['Link'],"image_url":element['image'],"pub_date":element['PubDate'],"description":element['Description'], "title":element['Title']})


# ### Adicionando features aos artigos

# ### Conectando áreas similares
# SE mais de uma categoria aparece no artigo, gerar conexões entre elas. Como tive que iterar por todas, na outra célula apago os `self-loops`

# In[ ]:


print('Features')
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


# ### Adicionando Super-Labels
# Agora com as `super-labels` adicionadas, crio a feature em cada nó de texto

# In[ ]:


# print('Labels')
# with driver.session() as sess:
#     for element in response:
#         if 'Label' in element.keys():
#             sess.run("""\
#                     MATCH (b:Text {title: $title})
#                     SET b.label = $label
#                     """, {"label":element['Label'],"title":element['Title']})


# ### Wikipedia
# Uso a api da wikipedia para gerar descrições nos nós de categorias, salvo o arquivo 'wiki.json' para não precisar fazer request denovo, eles são a parte mais demoradade desse código'

# In[ ]:


print('Wikipedia')
with open('data/wiki.json','r') as fp:
    pre_dict = json.load(fp)


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


with open('data/wiki.json', 'w') as fp:
    json.dump(descriptions, fp)


# In[ ]:


with driver.session() as sess:
    for element in descriptions.keys():
        sess.run("""                MATCH (b:INTEREST {name: $name})
                SET b.description = $description
                """, {"name":element,"description":descriptions[element]})
    


# In[ ]:



for key,value in descriptions.items():
    if value == 'Not Found':
        print(key)


# In[ ]:




