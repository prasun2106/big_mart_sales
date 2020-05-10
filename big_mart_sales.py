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
#  2. Data Cleaning
#  3. Univariate analysis
#  4. Bivariate analysis
#  5. Preprocessing
#  
#  
#  Preprocessing: [Tree-based and non-tree based models require different kind of preprocessing to be done]
#  1. Data cleaning and Missing Value treatment
#  1. Encoding categorical variables
#      * tree based model can work well with label encoding
#      * non tree based models (linear models, neural networks) need one hot encoding if there aren't any hierarchical categories.
#      * frequency encoding
#      * **Important**: one hot encoded features are always scaled. 
#      * Prefer label encoding over one hot for tree based models
#      * If a feature has many unique values, then use sparse matrix as there will be many zeros in one hot encoded format.
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

# In[102]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk


# ### * Exploratory Data Analysis Steps
# 1. Variable Identification

# In[103]:


train = pd.read_csv('Train_UWu5bXk.txt', sep = ',')


# In[104]:


test = pd.read_csv('Test_u94Q5KV.txt', sep = ',')


# In[105]:


print(train.shape)
print(test.shape)


# In[106]:


train.head()


# In[107]:


test.head()


# In[108]:


train.isnull().sum(axis = 1).sort_values(ascending = False)


# In[109]:


train.isnull().sum(axis = 0).sort_values(ascending = False)


# In[110]:


data = pd.concat([train, test], ignore_index=True)


# In[111]:


data.info()


# In[112]:


data.nunique().sort_values()


# #### Classification of Variables:
# We will classify the variables on two basis. The first basis of classification is whether they are string or numbers. This result has been presented above. The second classification is whether they are continuous or discrete. If they are continuous we will look at its mean, median, mode and visualize them by plotting histograms. If they are categorical, we will see the unique number of entries and then will plot the countplots.
# 
# #### Variable Identification results:
# 1. Categorical : 'Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Type', 'Outlet_Location_Type', 'Item_Identifier','Outlet_Identifier', 'Outlet_Establishment_Year'
# 2. Continuous : 'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales'
# 

# All features having unique values less than 20 are a categorical variables.

# ## * Exploratory Data Analysis
# 2. Data Cleaning and Missing Values Treatment

# In[113]:


data.columns


# In[114]:


for col in ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier','Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type']:
    print(col, data[col].unique())


# In[115]:


data.Item_Fat_Content.unique()


# We have to merge low fat, LF with Low Fat and reg with 'Regular'.

# In[116]:


data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(['reg'], 'Regular')
data.Item_Fat_Content.nunique()


# In[117]:


for col in data[['Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Type', 'Outlet_Location_Type' ]]:
    print(data[col].unique())


# In[118]:


data.isnull().sum()


# #### Comments:
# 
# 
# 1. Item_Outlet_Sales is the target variable. All the missing values correspond to the test file.
# 2. Item_Weight has 2439 and Outlet_Size has 4016 missing values.
# 3. Since, Outlet_Size has a lot of missing values, we need to decide whether we can drop this variable entirely.
# 4. We will impute Item_Weight with the mean of each item identifier

# In[119]:


data[['Item_Identifier','Item_Weight']].drop_duplicates().dropna()['Item_Identifier'].value_counts()


# The above result shows that each Identifier is mapped to only one weight. So we can safely replace nan with the item weights mapped to each item identifiers

# In[124]:


# get average weights for each identifier
avg_weight = data[['Item_Identifier','Item_Weight']].groupby('Item_Identifier').mean()
data = data.merge(avg_weight, how = 'left', on = 'Item_Identifier').drop('Item_Weight_x', axis = 1).rename(columns = {'Item_Weight_y':'Item_Weight'})


# In[ ]:


# imputing outlet size based on the 


# In[136]:


print(data[['Outlet_Size','Outlet_Identifier']].drop_duplicates())
print('\n')
print(data[['Outlet_Size','Outlet_Type']].drop_duplicates())


# Observation:
# * 3 of the outlets always have NaN values as Outlet Size
# * Grocery store is Nan or small, and Supermarket Type 1 is mapped to all possible values (small, medium, high, nan)
# * We can impute the missing values based on the mode of outlet size in each category of outlet_type

# In[170]:


mode_outlet_size = data.groupby(by = 'Outlet_Type').agg(lambda x: x.value_counts().index[0])['Outlet_Size']
mode_outlet_size = mode_outlet_size.reset_index()
print(mode_outlet_size)


# In[176]:


# Imputing missing outlet size
data = data.merge(mode_outlet_size, how = 'left', left_on = 'Outlet_Type', right_on = 'Outlet_Type')
data = data.drop('Outlet_Size_x', axis = 1)
data.rename(columns = {'Outlet_Size_y':'Outlet_Size'}, inplace = True)


# In[177]:


data.isnull().sum()


# ## * Exploratory Data Analysis
# 3. Univariate Analysis

# In[181]:


sns.distplot(data.Item_Weight)


# In[183]:


sns.countplot(data.Item_Fat_Content)


# In[184]:


sns.distplot(data.Item_Visibility, bins = 10)


# In[214]:


plt.figure(figsize= [25,12])
sns.set(style='darkgrid')
chart = sns.countplot(data.Item_Type, order = data.Item_Type.value_counts().index)
chart.set_xticklabels(chart.get_xticklabels(), rotation = 40)
plt.show()


# In[215]:


sns.distplot(data.Item_MRP)


# ## * Exploratory Data Analysis
# 4. Bivariate Analysis
# * continuous & continuous
# * continuous & categorical
# * categorical & categorical

# ### 1. continuous and continuous

# In[218]:


sns.scatterplot(x = data.Item_Weight, y = data.Item_Visibility)


# In[217]:


sns.pairplot(data)


# ### 2. Categorical and continuous
# 

# In[219]:


sns.boxplot(x = data.Item_Fat_Content, y = data.Item_Outlet_Sales)


# In[220]:


sns.boxplot(x = data.Item_Fat_Content, y = data.Item_MRP)


# In[242]:


sns.boxplot(x = data.Item_Fat_Content, y = data.Item_Visibility)


# In[ ]:





# In[224]:


def boxcompare(categorical, continuous):
    plt.figure(figsize=[20,6])
    sns.boxplot(x = categorical, y = continuous, data = data)
    


# In[225]:


boxcompare('Item_Type', 'Item_MRP')


# In[226]:


boxcompare('Item_Type', 'Item_Outlet_Sales')


# In[241]:



for col in [ 'Item_Fat_Content', 'Item_Type',
        'Outlet_Establishment_Year',
       'Outlet_Location_Type', 'Outlet_Type',  'Outlet_Size']:
    plt.figure()
    sns.countplot(x = data[col] )


# ### Conclusions of Univariate and Bivariate Analysis:
# 1. There are more low fat products
# 2. Fruits and Vegetables are the bestsellers
# 3. The maximum number of outlets are present in Tier 3
# 4. Most of the outlets are of small size.
# 5. The highest number of supermarkets are of Type 1.
# 6. Starchy foods are the most expensive ones on an average

# ### 5. Preprocessing
# 1. Data Cleaning - already done
# 2. Encoding

# In[245]:


data.info()


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

