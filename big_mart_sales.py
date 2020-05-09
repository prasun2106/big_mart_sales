#!/usr/bin/env python
# coding: utf-8

# # Project Description
# Link: https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/#About
# 
# BigMart has collected 2013 sales data for 1559 products across 10 stores located in different cities. Some attributes of both the store as well as the products have been provided. Our aim is to build a predictive model and find put the sales of each product at a particular store. Using this model, BigMart will try to understand the properties of store and product playing vital role in sales.

# ## Important Steps
# 
# * Hypothesis Generation
# * Data Exploration
# * Model Building

# ## * Hypothesis Generation
# 
# Without looking at the data, we will create our hypothesis listing all the possible factors affecting the sales of a product. There will be two types of hypthesis: store and product hypothesis.
# 
# ### Store Hypothesis:
# 1. City - Densely populated or more advanced cities would have higher sales.
# 2. Size - Large size of store implies higher sales as cutomers prefer to buy everything at one place.
# 3. Location - The store located in the more central part of the city where there is a good traffic are expected to have higher sales.
# 4. Opening Hours - Stores opening till late in night might have higher sales
# 5. Discounts - How frequently does the store offer a discount and sale?
# 6. Number of cash counters - Do people have to wait for billing?
# 7. Level of automation - Is there a good level of automation?
# 8. Staffs - How friendly are the staffs?
# 
# ### Product Hypothesis:
# 1. Type of Product - commodity or unique, daily use or luxury
# 2. Visibility - How much visible area does the product cover?
# 3. Packaging - How good is the packaging?
# 4. Health benefits - The products which are marketed as a healhy alternative to some other foods (e.g., brown bread, corn flakes) tends to have higher sales.
# 5. Advertising - Better advertising of product leads to higher sales.

# ## * Data Exploration
#  ### Essentially, there are three components of EDA. 
#  1. Understanding your variables
#  2. Data cleaning
#  3. Analyzing relationships between variables
#  
#  
#  ### Steps of data exploration and preprocessing:
#  
#  
#  Exploratory data analysis:
#  1. Variable Identification
#  2. Preprocessing
#  3. Univariate analysis
#  4. Bivariate analysis
#  
#  
#  Preprocessing: [Tree-based and non-tree based models require different kind of preprocessing to be done]
#  1. Data cleaning
#  1. Encoding categorical variables
#      * tree based model can work well with label encoding
#      * non tree based models (linear models, neural networks) need one hot encoding if there aren't any hierarchical categories.
#      * frequency encoding
#      * **Important**: one hot encoded features are always scaled. 
#      * Prefer label encoding over one hot for tree based models
#      * If a feature has many unique values, then use sparse matrix as there will be many zeros in one hot encoded format.
#  2. Missing Value treatment
#  3. Outlier Detection
#  4. Variable Transformation
#     * Normalization - Scaling
#         - tree based model doesn't depend on scaling
#         - non tree based models hugely depends on scaling
#         - choosing different scaling techniques (minmaxscaler and standardscaler of sklearn) might affect the model accuracy differently. It can act as one of the hyperparameters 
#     * Rank Transformation
#          - Rank transformation sets proper spaces between assorted values. In case of outliers, rank transformation gives better results as it moves outliers closer to other objects
#     * Log transformation or raising to a power < 1
#         - These kind of transformation moves large values closer and make values near to 0 more distinguishable
#  4. Variable Creation (aka Feature engineering, feature creation)
#      * based on prior knowledge
#      * based on EDA
#          - examples : add fractional part of prices as an additional column. helps model to understand people's perception in rounding off the prices
#          - mmultiply, divide, add, subtract to create new features which make more sense
#          - **Important** : In case of categorical features, we can concatenate two categorical variables and then one hot encode it to get better results
#      * Datetime
#          - Periodicity
#          - Time since certain event
#          - Differences between dates
#      * Coordintes
#          - Distance of interesting places from our data
#          - Center of clusters
#          - Aggregated statistics (popularity etc)
#  6. Sometimes concatenating two dataframes having different preprocessing helps in increasing model accuracy of non tree based models

# ## * Model Building

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk


# ### * Exploratory Data Analysis Steps
# 1. Variable Identification

# In[58]:


train = pd.read_csv('Train_UWu5bXk.txt', sep = ',')


# In[59]:


test = pd.read_csv('Test_u94Q5KV.txt', sep = ',')


# In[60]:


print(train.shape)
print(test.shape)


# In[61]:


train.head()


# In[62]:


test.head()


# In[63]:


train.isnull().sum(axis = 1).sort_values(ascending = False)


# In[64]:


train.isnull().sum(axis = 0).sort_values(ascending = False)


# In[65]:


data = pd.concat([train, test], ignore_index=True)


# In[66]:


data.info()


# #### Classification of Variables:
# We will classify the variables on two basis. The first basis of classification is whether they are string or numbers. This result has been presented above. The second classification is whether they are continuous or discrete. If they are continuous we will look at its mean, median, mode and visualize them by plotting histograms. If they are categorical, we will see the unique number of entries and then will plot the countplots.
# 

# In[71]:


data.nunique().sort_values()


# All features having unique values less than 20 are a categorical variables.

# ## 1. Preliminary Cleaning

# In[42]:


data.columns


# In[53]:


for col in ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier','Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type']:
    print(col, data[col].unique())


# In[54]:


data.Item_Fat_Content.unique()


# We have to merge low fat, LF with Low Fat and reg with 'Regular'.

# In[55]:


data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(['reg'], 'Regular')
data.Item_Fat_Content.nunique()


# In[16]:


for col in train[['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Type', 'Outlet_Location_Type' ]]:
    print(train[col].unique())


# # Variable Identification result:
# 1. Categorical : 'Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Type', 'Outlet_Location_Type', Item_Identifier,'Outlet_Identifier', 'Outlet_Establishment_Year'
# 2. Continuous : 'Item_Weight', 'Item_Visibility', 'Item_MRP', 

# In[17]:


train.columns


# ## 2. Univariate Analysis

# In[18]:


train['Item_Weight2'] = train['Item_Weight'].fillna(train['Item_Weight'].mean())


# In[19]:


sns.distplot(train.Item_Weight2, bins = 10)


# In[20]:


train['Item_Weight'] = train['Item_Weight2']


# In[21]:


train =  train.drop('Item_Weight2', axis = 1)


# In[22]:


train.columns


# In[23]:


sns.countplot(train.Item_Fat_Content)


# In[24]:


sns.distplot(train.Item_Visibility, bins = 10)


# In[25]:


plt.figure(figsize= [25,6])
sns.countplot(train.Item_Type, orient =30)


# In[26]:


train.columns


# In[27]:


sns.distplot(train.Item_MRP)


# # 3. Bivariate Analysis
# 1. continuous continuous
# 2. continuous categorical
# 3. categorical categorical

# In[28]:


train.columns


# Hypothesis : Light wright items should have higher visibility

# In[29]:


sns.scatterplot(x = train.Item_Weight, y = train.Item_Visibility)


# In[30]:


sns.pairplot(train)


# ### 2. Categorical and continuous
# 

# In[31]:


train.columns


# In[32]:


sns.boxplot(x = train.Item_Fat_Content, y = train.Item_Outlet_Sales)


# In[33]:


sns.boxplot(x = train.Item_Fat_Content, y = train.Item_MRP)


# In[34]:


sns.boxplot(x = train.Item_Fat_Content, y = train.Item_Visibility)


# In[35]:


def boxcompare(categorical, continuous):
    plt.figure(figsize=[20,6])
    sns.boxplot(x = categorical, y = continuous, data = train)
    


# In[36]:


boxcompare('Item_Type', 'Item_MRP')


# In[37]:


boxcompare('Item_Type', 'Item_Outlet_Sales')


# We should combine both the train and test sets together before implementing feature engineering. This prevents us from doing the same work twice. In the data exploration step, we have already covered variable identification, univariate analysis, and bivariate analysis. Now we will move on to the remaining steps i.e., Missing Values Imputation, Outliers Handling, Feature Engineering and Feature Normalization.

# In[38]:


train['Source'] = 'train'
test['Source'] = 'test'


# In[39]:


data = pd.concat([train, test], ignore_index= True)


# In[40]:


train.shape, test.shape, data.shape


# In[41]:


data.apply(lambda x : sum(x.isnull()))


# In[17]:


data.apply(lambda x : x.nunique())


# In[43]:


categorical_columns = [x for x in data.dtypes.index if data.dtypes[x] == 'object']


# In[44]:


categorical_columns2 = [x for x in data.columns if data.dtypes[x] == 'object']


# In[45]:


categorical_columns2


# In[46]:


for col in ('Item_Identifier', 'Outlet_Identifier', 'Source'):
     categorical_columns2.remove(col)


# In[47]:


categorical_columns = categorical_columns2


# In[48]:


categorical_columns


# In[49]:


for col in categorical_columns:

    plt.figure(figsize = [20,6])
    sns.countplot(x = train[col] )


# # Conclusions:
# 1. There are more low fat products
# 2. Fruits and Vegetables are the best sellers
# 3. The maximum number of outlets are present in Tier 2
# 4. Most of the outlets are of medium size.
# 5. The highest number of supermarkets are of Type 1.

# # 4. Missing Value Treatment

# In[50]:


# Finding Missing Values

data.apply(lambda x : sum(x.isnull()))


# 1. Item_Outlet_Sales is the target variable. All the missing values correspond to the test file.
# 2. Item_Weight has 976 and Outlet_Size has 4016 missing values.
# 3. Since, Outlet_Size has a lot of missing values, we need to decide whether we can drop this variable entirely.

# In[51]:


# Visualizing the missing values:

sns.heatmap(pd.isnull(data),cmap = 'Blues')


# In[52]:


# Imputing the Missing Values: Item Weight is a numerical variable. So we will impute the NaNa with the mean value.
data['Item_Weight'][data['Item_Weight'].isnull()] = data['Item_Weight'].mean(skipna = True)


# In[53]:


data['Outlet_Size'][data['Outlet_Size'].isnull()] = data['Outlet_Size'].mode()


# In[54]:


sum(data['Outlet_Size'].isnull())


# In[55]:


data['Outlet_Size'][data['Outlet_Size'].isnull()]


# In[56]:


data['Outlet_Size'].mode(dropna = True)


# In[57]:


data.Outlet_Size[data.Outlet_Size.isnull()] = data.Outlet_Size.mode()[0]


# In[58]:


sns.heatmap(data.isnull())


# ### We have succesfully imputed all the missing values.

# # 5. Outlier Detection

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # 6. Feature Engineering
# ## I. Variable Creation

# In[59]:


data.columns


# In[60]:


for col in categorical_columns2:
    print(data[col].unique())


# # Ideas for feature engineering:
# 1. Further divide the Item_Type in healthy, unheathy and neutral food items.
# 2. Create a variable based on the Item_Type which has three categories : food, drinks, non-consumables
# 3. Modify Item_Visibility
# 4. Modify categories of Item_Fat_Content
# 5. Create a variable indicating years of operation of each store

# In[61]:


for i in range(len(data['Item_Type'])):
    if data.Item_Type in 
    data['Item_Type_2'] = data['Item_Type']
    


# In[62]:


for col in categorical_columns2:
    print( col, data[col].unique())
    print('\n')


# # 1. Item type modified to healthy, non-healthy and neutral

# In[63]:


Item_Type_Modified = ['Dairy', 'Healthy', 'Soft Drinks', 'Unhealthy', 'Meat', 'Healthy', 'Fruits and Vegetables', 'Healthy', 'Household',
'Neutral', 'Baking Goods', 'Unhealthy', 'Snack Foods','Unhealthy', 'Frozen Foods', 'Unhealthy',
'Breakfast', 'Healthy', 'Health and Hygiene', 'Neutral',  'Hard Drinks', 'Unhealthy', 'Canned', 'Unhealthy',
'Breads', 'Unhealthy', 'Starchy Foods', 'Unhealthy', 'Others', 'Neutral', 'Seafood', 'Healthy']


# In[64]:


data['Item_Type_2'] = data.Item_Type.copy()


# In[65]:


data.Item_Type_2 = data.Item_Type_2.replace({'Dairy' : 'Healthy', 'Soft Drinks' : 'Unhealthy', 'Meat' : 'Healthy', 'Fruits and Vegetables' : 'Healthy', 'Household' : 'Neutral', 'Baking Goods' : 'Unhealthy', 'Snack Foods' : 'Unhealthy', 'Frozen Foods' : 'Unhealthy', 'Breakfast' : 'Healthy', 'Health and Hygiene' : 'Neutral', 'Hard Drinks' : 'Unhealthy', 'Canned' : 'Unhealthy', 'Breads' : 'Unhealthy', 'Starchy Foods' : 'Unhealthy', 'Others' : 'Neutral', 'Seafood' : 'Healthy'})


# In[66]:


data.Item_Type_2.unique()


# # 2. Food, Drinks and Non-Consummables

# In[67]:


data['Item_Type_3'] = data.Item_Type.copy()


# In[68]:


data.Item_Type_3 = data.Item_Type_3.replace({'Dairy' : 'Food', 'Soft Drinks' : 'Drinks', 'Meat' : 'Food', 'Fruits and Vegetables' : 'Food', 'Household' : 'Non-Consummables', 'Baking Goods' : 'Food', 'Snack Foods' : 'Food', 'Frozen Foods' : 'Food', 'Breakfast' : 'Food', 'Health and Hygiene' : 'Non-Consummables', 'Hard Drinks' : 'Drinks', 'Canned' : 'Food', 'Breads' : 'Food', 'Starchy Foods' : 'Food', 'Others' : 'Non-Consummables', 'Seafood' : 'Food'})


# In[69]:


data["Item_Type_3"].unique()


# # 3. Modify Item_Visibility

# In[70]:


data.Item_Visibility.head()


# In[71]:


data.Item_Visibility[data.Item_Visibility == 0]


# In[72]:


data['Item_Visibility'] = data.Item_Visibility.replace(to_replace = 0, value = np.nan)


# In[73]:


data['Item_Visibility'] = data['Item_Visibility'].fillna(data.Item_Visibility.mean(skipna=  True))


# In[74]:


pd.DataFrame(data.Item_Visibility).describe()


# Item having 0 visibility have been replaced by the average of other visibilities.

# # 4. Modifying Item_Fat_Content
# We can see that corresponding to Non-Consummables we have some Item_Fat_Content values which is illogical. We will replace these entries with 'Non-Consummables'.

# In[75]:


data[data['Item_Type_3'] == 'Non-Consummables'].head()


# In[76]:


data.loc[data['Item_Type_3'] == 'Non-Consummables','Item_Fat_Content' ] = 'Non-Consummables'


# In[77]:


data[data['Item_Type_3'] == 'Non-Consummables'].head()


# # 5. Year of Operation

# In[78]:


data['Years_Of_Operation'] = 2019 - data.Outlet_Establishment_Year


# In[79]:


data.head()


# 
# ## II. Variable Transformation
# Variable transformation mainly involves normalization and encoding. 
# We wish to see the effects of feature normalization on the accuracy of the model. For this reason, we are keeping the feature normalization for the later part of the exeercise. Since sklearn only accepts numerical values, we will use one hot encoding to convert string categorical variable into numerical categorical variable.

# In[98]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:





# In[100]:


categorical_columns = [x for x in data.columns if data.dtypes[x] == 'object']


# In[101]:


categorical_columns


# In[102]:


data_copy = data.copy()


# In[103]:


encoder = OneHotEncoder(categorical_columns)


# In[107]:


data_copy = OneHotEncoder.fit_transform(data_copy, categorical_columns).toarray()


# In[108]:


data_encoded = data.copy()


# In[ ]:


data

