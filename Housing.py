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


# In[24]:


housing = load_housing_data()
housing.head(4862)


# In[4]:


housing.info()


# In[5]:


housing["ocean_proximity"].value_counts()


# In[6]:


housing.describe()


# In[27]:


import matplotlib.pyplot as plt 
housing.hist(bins=50, figsize=(25,20)) 
plt.show()


# In[8]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# In[ ]:


# as the median_income feature has wide range of values, when it is splitted it might cause bias for the ML algo
# so we are doing feature engineering, i.e creating a new feature from the existing feature
# And making that new feature balanced with better representation of the values to avoid bias


# In[25]:


import numpy as np
housing['income_cat'] = np.ceil(housing['median_income']/1.5)

print(housing['income_cat'].head(4862))


# In[37]:


import numpy as np
# creating a new column named income_cat by downscaling the median income by the value of 1.5
# No idea how the value 1.5 is chosen to scaledown probably bcz the max value is 15
housing['income_cat'] = np.ceil(housing['median_income']/1.5) 
# np.ceil is a rounding function in numpy 
# It rounds the number to the largest closest number as a float like 1.7 -> 2.0 ; 5.3 -> 6.0

# now let's see the histogram of income_cat
housing['income_cat'].hist(bins = 50, figsize = (5,5))
plt.xticks(ticks=np.arange(0, 16, 1))
plt.show()


# In[59]:


# in the histogram of incom_cat most of the income values are clustered between 1 - 5 but still some go beyond 6
# So the samples for the income category greater than 5 (6 - 11) are less this can lead to a bias in stratified sampling 
# i.e the categories (6-11) has less representation than other categories

# Hence we merge the categories 6-11 to the category 5
housing.loc[housing['income_cat'] >= 5, 'income_cat'] = 5.0
# The syntax in the above line is split into three parts
# 1. Selecting the values greater than or equal to 5:  housing['income_cat'] >= 5

# 2. choosing the rows and columnn: housing.loc[housing['income_cat'] >= 5, 'income_cat'] 
# --> i.e for the rows in the housing['income_cat'] >= 5 in the column 'income_cat' in housing dataframe

# 3. assigning those listed values as 5.0: housing.loc[housing['income_cat'] >= 5, 'income_cat'] = 5.0

# Now let's again look at the income_cat in which the categories (6-11) are merged to 5
housing['income_cat'].hist(bins = 10, figsize = (5,5))
plt.xticks(ticks=np.arange(0, 16, 1))
plt.show()


# In[60]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing['income_cat'].value_counts() / len(housing)


# # add comments that we have to sratified sample based on target
