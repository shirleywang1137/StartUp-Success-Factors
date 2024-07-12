#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import modules
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# All imports you likely would need
## Models and modeling
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

## Data Munging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer

## Measurements
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay # for newer versions of sklearn
from sklearn.metrics import plot_confusion_matrix  # for older versions of sklearn
import matplotlib.pyplot as plt


# In[2]:


# read data
df = pd.read_csv("startup data.csv", index_col=0)
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


# data wrangling
df = df.drop(['Unnamed: 6'], axis=1)
df = df.drop(['state_code.1'], axis=1)
df = df.drop(['object_id'], axis=1)
start = ['c:']
end = ['']
df['id'] = df['id'].replace(start, end, regex=True)
df.avg_participants = df.avg_participants.round(4)
df['age_first_milestone_year'] = df['age_first_milestone_year'].fillna(0)
df['age_last_milestone_year'] = df['age_last_milestone_year'].fillna(0)


# In[5]:


# determine age of startup
df['closed_at'] = pd.to_datetime(df['closed_at'])
df['founded_at'] = pd.to_datetime(df['founded_at'])

# too many NaN in age
df["age"] = (df["closed_at"]-df["founded_at"])
df["age"]=round(df.age/np.timedelta64(1,'Y'))


# In[6]:


# variable modification
df['status'] = df.status.map({'acquired':1, 'closed':0})
df['status'].astype(int)

#has rounds of funding
df['has_rounds'] = np.where((df['has_roundA'] == 1) | (df['has_roundB'] == 1) | (df['has_roundC'] == 1) | (df['has_roundD'] == 1), 1, 0)

#has investor
df['has_investor'] = np.where((df['has_VC'] == 1) | (df['has_angel'] == 1), 1, 0)


# In[7]:


# probability analysis
""" goal: narrow down a few factors that show statistically significant results
--> later, we will use them as parameters for machine learning model"""


# In[8]:


# visualizations
""" goal: explore startup success by different key factors:
--> location
--> funding amount
--> industry
--> age"""


# In[9]:


# success by industry
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x="category_code", hue="status", data=df,
              order=df.category_code.value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set(xlabel="Industry", ylabel="Num of Startups")
plt.legend(bbox_to_anchor=(0.93, 0.93))


# In[10]:


# success by location per industry
loc_industry_success = df.groupby(['state_code','category_code']).size().rename('total_startups').reset_index()
loc_industry_success = loc_industry_success[loc_industry_success.groupby('state_code')['total_startups'].transform(max) == loc_industry_success['total_startups']]
loc_industry_success = loc_industry_success.sort_values('total_startups', ascending=False)
loc_industry_success.head(10)


# In[11]:


# funding rounds by location
rounds_df = df.sort_values(by=["funding_rounds"], ascending=True)
sns.catplot(data=rounds_df, y="state_code", x="funding_rounds", kind="bar")


# In[12]:


# funding amounts by location
loc_funding_amt = df.groupby(['state_code','funding_total_usd']).size().rename('total_funding').reset_index()
loc_funding_amt = loc_funding_amt[loc_funding_amt.groupby('state_code')['total_funding'].transform(max) == loc_funding_amt['total_funding']]
loc_funding_amt = loc_funding_amt.sort_values('total_funding', ascending=False)
loc_funding_amt.head(10)


# In[29]:


# funding amounts by industry
cat_funding_amt = df.groupby(['category_code','funding_total_usd']).size().rename('total_funding').reset_index()
cat_funding_amt = cat_funding_amt[cat_funding_amt.groupby('category_code')['total_funding'].transform(max) == cat_funding_amt['total_funding']]
cat_funding_amt = cat_funding_amt.sort_values('total_funding', ascending=False)
cat_funding_amt.head(10)


# In[16]:


# average participants
sns.boxplot(x = "has_investor", y = "avg_participants", 
            data = df, hue = "status")


# In[17]:


# milestones
sns.histplot(x = "milestones", data = df, hue="status")


# In[18]:


# relationships
sns.scatterplot(data=df, x='age', y='relationships', hue='status')


# In[26]:


# pair plot - can use to show lack of correlation
pair_data = df[['age','milestones','relationships', 'status']]
sns.pairplot(pair_data, hue="status")


# In[20]:


# more correlational data
heatmap_data = df[['age', 'milestones','relationships','status']]
sns.heatmap(heatmap_data.corr(), annot=True)


# In[28]:


# binary vars heatmap
heatmap_data2 = df[['has_investor', 'has_rounds','is_top500','status']]
sns.heatmap(heatmap_data2.corr(), annot=True)


# In[21]:


# machine learning model
""" goal: use top 3-4 factors from prob & visualization analysis
--> create a machine learning model to predict startup success"""


# In[22]:


df_ml = df[['status', 'funding_total_usd', 'milestones', 'state_code']]
col_target = 'status'

target = df_ml[col_target].values
categorical = OneHotEncoder().fit_transform(df_ml[['state_code']].values).toarray()
data = np.append(categorical, df_ml[['funding_total_usd', 'milestones']].values, axis=1)

train_data, test_data, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=216)

params_to_try = {'n_neighbors': range(1, 20)}
knn_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params_to_try)
knn_search.fit(train_data, train_target)

baseline_model = DummyClassifier(strategy="most_frequent")
baseline_model.fit(X=test_data, y=test_target)
baseline_predicted = baseline_model.predict(data)

accuracy_model = knn_search.score(test_data, test_target)
accuracy_baseline = accuracy_score(y_true=target, y_pred=baseline_predicted)
print('best n:', knn_search.best_params_['n_neighbors'])

ConfusionMatrixDisplay.from_estimator(estimator=knn_search, X=data, y=target)
plt.grid(False)
ConfusionMatrixDisplay.from_estimator(estimator=baseline_model, X=data, y=target)
plt.grid(False)


# In[23]:


print(df_ml.shape)
print('>=300 rows?', df_ml.shape[0] >= 300)
print('Categories:')
print(df_ml[col_target].value_counts())
print('Accuracy: ', accuracy_model)
print('Accuracy Baseline: ', accuracy_baseline)


# In[ ]:




