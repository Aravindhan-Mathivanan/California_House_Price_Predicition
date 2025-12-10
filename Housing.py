#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tarfile
from six.moves import urllib

Download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
Housing_path = "datasets/housing"
Housing_url = Download_root + Housing_path + "/housing.tgz"

def fetch_housing_data(housing_url = Housing_url, housing_path = Housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()


# In[2]:


import pandas as pd
def load_housing_data(housing_path=Housing_path): 
    csv_path = os.path.join(housing_path, "housing.csv") 
    return pd.read_csv(csv_path)


# In[3]:


housing = load_housing_data()
housing.tail(2000)


# In[4]:


housing.info()


# In[5]:


housing["ocean_proximity"].value_counts()


# In[6]:


housing.describe()


# In[22]:


import matplotlib.pyplot as plt 
housing.hist(bins=50, figsize=(25,20)) 
plt.show()


# In[24]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# In[ ]:




