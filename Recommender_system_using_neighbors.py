#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import pairwise_distances 


# In[2]:


dataset = pd.read_csv('C:/Users/sam/Desktop/Kaggle/people_wiki/people_wiki.csv',index_col='name')


# In[3]:


data=dataset['text']


# In[4]:


data.head()


# In[5]:


location=data.index.get_loc('Barack Obama')


# #####  Getting TFID

# In[11]:


dataTf = TfidfVectorizer(stop_words='english').fit_transform(data)


# In[ ]:





# ### Define function using default distance

# In[12]:


def get_similar_people(name,neighbors):
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(dataTf)
    location=data.index.get_loc(name)
    distance,indices = nbrs.kneighbors(dataTf[location])
    people = pd.Series(indices.flatten()).map(data.reset_index()['name'])
    neighbors = pd.DataFrame({'distance':distance.flatten(),'people':people})
    return neighbors


# In[ ]:





# In[15]:


get_similar_people('Barack Obama',7)


# In[ ]:





# ### Define function using cosine similaritie

# In[16]:


def get_similar_people(name,neighbors):
    nbrs = NearestNeighbors(n_neighbors=neighbors,metric='cosine').fit(dataTf)
    location=data.index.get_loc(name)
    distance,indices = nbrs.kneighbors(dataTf[location])
    people = pd.Series(indices.flatten()).map(data.reset_index()['name'])
    neighbors = pd.DataFrame({'distance':distance.flatten(),'people':people})
    return neighbors


# In[ ]:





# In[17]:


get_similar_people('Barack Obama',7)


# In[ ]:





# In[ ]:





# In[ ]:




