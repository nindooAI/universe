
# coding: utf-8

# # Discover DB
# 

# ## Imports e configurações iniciais

# In[22]:


import pandas as pd
import os
import requests
import json
from neo4j import GraphDatabase, basic_auth
import wikipedia


# ## Env Variables

# In[23]:


url = 'https://spxa6xmc58.execute-api.us-west-2.amazonaws.com/prod/'
n4j_pass = "nindoo123"
n4j_login = 'neo4j'


# ## Data loading

# In[24]:


def get_crawler(url):
    response = requests.get(url)
    
    return response.text
response = json.loads(get_crawler(url))
#print(response)


# ## Descobrimento de dados

# In[25]:


#soma = 0
for element in response:
    print(element.keys())
    #print(element['Category'])
    # print(element['Description'])
    # print(element['Link'])
    # print(element['PubDate'])
    #print(element['Title'])
    # print(element['image'])
    break


# ## Conectando ao db

# In[26]:


driver = GraphDatabase.driver( "bolt://localhost:7687",  auth=basic_auth(n4j_login,n4j_pass))
sess = driver.session()


# ### Interesses:

# In[27]:


with driver.session() as sess:
    for element in response:
        for interest in element['Category']:
            if 'Label' in element.keys():
                sess.run("""                    MERGE (a:INTEREST {name: $name})
                    SET a.label = $label
                    """, {"name":interest,"label":element['Label']})
            else:
                sess.run("""                    MERGE (a:INTEREST {name: {name}})
                    """, {"name":interest})


# ## Artigos ligados à Interesses

# In[28]:


with driver.session() as sess:
    for element in response:
        sess.run("""            MERGE (b:Text {title: {title}})
            """, {"title":element['Title']})


# In[29]:


with driver.session() as sess:
    for element in response:
        for interest in element['Category']:
            sess.run("""                MATCH (a:INTEREST {name: {name}}),(b:Text {title: {title}})
                MERGE (b)-[r:BELONGS_TO]->(a)
                """, {"name":interest, "title":element['Title']})


# In[30]:


with driver.session() as sess:
    for element in response:
        sess.run("""                MATCH (b:Text {title: {title}})
                SET b.link = {link}, b.image_url = {image_url}, b.description = {description},                    b.date = {pub_date}
                SET b:Text:Blog
                """, {"link":element['Link'],"image_url":element['image'],"pub_date":element['PubDate'],"description":element['Description'], "title":element['Title']})


# ### Adicionando features aos artigos

# ### Conectando áreas similares

# In[31]:


with driver.session() as sess:
    for element in response:
        for interest in element['Category']:
            if len(element['Category']) != 1:    
                sess.run("""                    MATCH (a:INTEREST {name: {name}}), (b:INTEREST {name: {other}}) 
                    WHERE NOT (a)-[:IS_RELATED]-(b) 
                    MERGE (a)-[:IS_RELATED]->(b)
                    """, {"name":element['Category'][0], "other":interest})


# In[32]:


### Deletando ligações iguais
with driver.session() as sess:
    for element in response:
        for interest in element['Category']:
            if len(element['Category']) >=1:    
                sess.run("""                    MATCH (a:INTEREST {name: {name}})-[r:IS_RELATED]->(b:INTEREST {name: {other}}) 
                    WHERE a.name = b.name
                    DELETE r
                    """, {"name":element['Category'][0], "other":interest})


# ### Adicionando Super-Labels

# In[33]:


with driver.session() as sess:
    for element in response:
        if 'Label' in element.keys():
            sess.run("""                    MATCH (b:Text {title: {title}})
                    SET b.label = {label}
                    """, {"label":element['Label'],"title":element['Title']})


# ### Wikipedia

# In[34]:


with open('data/wiki.json','r') as fp:
    pre_dict = json.load(fp)


# In[35]:


lista = []
for element in response:
        for interest in element['Category']:
            if interest not in pre_dict.keys():
                lista.append(interest)
interests = set(lista)
print(len(lista))
print(len(interests))
descriptions = pre_dict
for element in interests:
    if element not in descriptions.keys():
        try:
            pagina = wikipedia.page(element)
            descriptions[element] = pagina.summary
        except:
            descriptions[element] = 'Not Found'
        


# In[36]:


with open('data/wiki.json', 'w') as fp:
    json.dump(descriptions, fp)


# In[37]:


with driver.session() as sess:
    for element in descriptions.keys():
        sess.run("""                MATCH (b:INTEREST {name: {name}})
                SET b.description = {description}""", 
                {"name":element,"description":descriptions[element]})
    


# In[38]:


for key,value in descriptions.items():
    if value == 'Not Found':
        print(key)

