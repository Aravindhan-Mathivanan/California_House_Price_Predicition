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
housing.head(4862)


# In[4]:


housing.info()


# In[5]:


housing["ocean_proximity"].value_counts()


# In[6]:


housing.describe()


# In[7]:


import matplotlib.pyplot as plt 
housing.hist(bins=50, figsize=(25,20)) 
plt.show()


# In[8]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# In[9]:


# as the median_income feature has wide range of values, when it is splitted it might cause bias for the ML algo
# so we are doing feature engineering, i.e creating a new feature from the existing feature
# And making that new feature balanced with better representation of the values to avoid bias


# In[10]:


## So the point of creating new feature is the StratifedShuffleSplit function requieres a categorical feature to keep the same percentage of  
# samples in both training and test set

## as the dataset has almost unique numerical values for each samples for each features (except: Ocean_proximity a categorical feature)
## A categorical feature that has balanced no of samples is required to do stratified sampling

## Why they ignored Ocean_proximity?
# As in the previous snippet we checked the value counts of ocean proximity in which the island category has only 5 samples 
# which can induce a bias in the split


# In[11]:


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


# In[12]:


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


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing['income_cat'].value_counts() / len(housing)*100
# checking how much percentage of samples fall into each category of income_cat feature


# In[14]:


# So as we tried stratified sampling based on a new feature, let's drop it as we no longer need it
for set in (strat_train_set, strat_test_set):
    set.drop(['income_cat'], axis = 1, inplace = True)


# In[15]:


# Visualizing the data to get more insights
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude')
# this plot shows the houses location in california housing dataset
# The plot quite resembles the map of california


# In[16]:


# Let's the density of houses by setting the alpha as 0.1 (the alpha sets transparency of each points)
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.1)
# Now we can se the places with more density darker (because there are more data points makes it more darker)


# In[17]:


# Let's have a better visualization of more features together
housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.3, s = housing['population']/100, label = 'population',
             c = 'median_house_value', cmap=plt.get_cmap('rainbow'),colorbar = 'True')
# Let's see what's happening here step by step
# step 1: sets up x and y
# Creates an empty plot with:
# kind of the plot is set to scatter
# X-axis: longitude values
# Y-axis: latitude values
# Result: A map of California with dots at correct locations

# step 2: adjust size for each points of scatter plot
# For EACH dot:
# - Check population value
# - Divide by 100
# - Set dot size accordingly
# Example: 
# House 1: Population 5000 → Dot size = 50 pixels
# House 2: Population 200 → Dot size = 2 pixels

# step 3: Apply colour
# For EACH dot:
# - Check house price
# - Look up color in 'jet' colormap
# - Apply that color to dot
# Example (jet colormap):
# $100,000 → Dark blue
# $300,000 → Green  
# $500,000 → Yellow
# $700,000 → Red

# step 4: Apply transparency
# Make ALL dots 40% transparent
# So overlapping dots show through

# Now the maps looks more good with each dots representing the location of houses (x and y), size of dots representing population in each houses
# the color of each dots represents the population and the colourbar represents the population that the colour indicates


# In[18]:


#This plot tells that the housing prices are very much related to the location (e.g., close to the ocean) and to the population density


# In[21]:


## Since the dataset is not too large, you can easily compute the standard correlation coefficient (also called Pearson’s r) between every pair of
# attributes using the corr() method
# As the ocean_proximity is a categorical feature it cannot be correlated with other features so let's drop it temporarily

housing_temp = housing.drop('ocean_proximity', axis = 1)
corr_matrix = housing_temp.corr()
corr_matrix['median_house_value'].sort_values(ascending = 0)


# In[ ]:


## The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation; 
# for example, the median house value tends to go up when the median income goes up. 
# When the coefficient is close to –1, it means that there is a strong negative correlation;
# you can see a small negative correlation between the latitude and the median house value
# coefficients close to zero mean that there is no linear correlation.


# In[41]:


# Another way to check the correlation between attributes is by using Pandas Scatter_matrix function 
# Which plots each attriubute agains all other attributes in the dataset
# As the dataset has 11 attributes it'll produce 11*11 = 121 plots so let's plot only most correlated attributes with median_housing_value

from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize = (10,7), color = 'blue')


# In[103]:


# In the plots the diagonal elements are each feature plotted against itself but that would result in a straight line
# Instead pandas shows the histogram of the respective attribute

# From these plots the most promising attribute to predict the median_house_value is median_income

housing.plot(kind = 'scatter', x = 'median_income', y = 'median_house_value', alpha = 0.5, color = 'darkviolet')


# In[104]:


'''This plot reveals a few things. First, the correlation is indeed very strong;
 you can clearly see the upward trend and the points are not too dispersed. 
 Second, the price cap that we noticed earlier is clearly visible as a horizontal line at $500,000. 
 But this plot reveals other less obvious straightlines: a horizontal line around $450,000, 
 another around $350,000, perhaps one around $280,000, and a few more below that. You may want to try
 removing the corresponding districts to prevent your algorithms from learning to reproduce these data quirks.'''


# In[105]:


# Experimenting with attribute combination


# In[48]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"] 
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"] 
housing["population_per_household"]=housing["population"]/housing["households"]


# In[49]:


housing.head()


# In[ ]:




