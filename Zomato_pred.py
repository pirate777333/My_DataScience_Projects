import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from math import sqrt
from scipy.stats import boxcox
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

def cleanrate(text):
    x=str(text).split('/5')[0]
    x=x.strip()
    return x

def cleanapprox_cost(text):
    x=str(text).replace(',','')
    return x

def assign(text):
    if text>0:
        return 1
    else:
        return 0

def thresh(text):
    threshold=3.75
    if text>threshold:
        return 1
    else:
        return 0

df=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project93/zomato.csv')
df.drop(['url'],axis=1,inplace=True)
df.drop(['address'],axis=1,inplace=True)
df.drop(['phone'],axis=1,inplace=True)
df.drop(['reviews_list'],axis=1,inplace=True)
df.drop(['menu_item'],axis=1,inplace=True)
df.drop(['dish_liked'],axis=1,inplace=True)

df['approx_cost(for two people)']=df['approx_cost(for two people)'].apply(cleanapprox_cost)
df['approx_cost(for two people)']=pd.to_numeric(df['approx_cost(for two people)'], errors ='coerce')
df['approx_cost(for two people)'].fillna((df['approx_cost(for two people)'].mean()), inplace=True)

df["rate"].replace({"NEW": np.nan, "-": np.nan}, inplace=True)
df['rate']=df['rate'].apply(cleanrate)
df['rate']=pd.to_numeric(df['rate'], errors ='coerce')
df['rate'].fillna((df['rate'].mean()), inplace=True)

df['rest_type'].fillna((df['rest_type'].mode()[0]), inplace=True)
df['cuisines'].fillna((df['cuisines'].mode()[0]), inplace=True)

df.dropna(inplace=True)

print(df.head())
print(df.columns)
print(df.dtypes)
print(df.tail())
print(df.shape)
print(df.describe())
print(df.isnull().sum())
print(df.rate.unique())
#print(df['approx_cost(for two people)'].unique())

sns.countplot(df['rest_type'])
plt.show()

resttypes=df.groupby('rest_type')['cuisines'].count().reset_index()
resttypes=resttypes.sort_values(by='cuisines',ascending=False)
fig=px.pie(resttypes.iloc[:10,:], values='cuisines', names='rest_type',
           title='Restaurant Types', template='plotly_dark')
fig.show()

rates=df.groupby('name')['rate'].max().reset_index()
rates=rates.sort_values(by='rate',ascending=False)
fig=px.bar(rates.iloc[:10,:], x='name', y='rate',
           title='Highest rated', template='plotly_dark')
fig.show()

voted=df.groupby('name')['votes'].sum().reset_index()
voted=voted.sort_values(by='votes',ascending=False)
fig=px.bar(voted.iloc[:10,:], x='name', y='votes',
           title='Highest voted', template='plotly_dark')
fig.show()

costs=df.groupby('name')['approx_cost(for two people)'].max().reset_index()
costs=costs.sort_values(by='approx_cost(for two people)',ascending=False)
fig=px.bar(costs.iloc[:10,:], x='name', y='approx_cost(for two people)',
           title='Highest Cost', template='plotly_dark')
fig.show()

costs=df.groupby('name')['approx_cost(for two people)'].max().reset_index()
costs=costs.sort_values(by='approx_cost(for two people)',ascending=True)
fig=px.bar(costs.iloc[:10,:], x='name', y='approx_cost(for two people)',
           title='Highest Cost', template='plotly_dark')
fig.show()

sns.countplot(df['online_order']) # or groupby > count or value_count()
plt.show()

sns.countplot(df['book_table'])
plt.show()

budget=df.groupby(['location','name'])['approx_cost(for two people)'].min().reset_index()
#print(budget.location.unique())

sns.countplot(train_val_rest['target'])
plt.show()

corr=train_val_rest.corr()
sns.heatmap(corr,annot=True)
plt.show()

df.drop(['name'],axis=1,inplace=True)
df.drop(['listed_in(type)'],axis=1,inplace=True)
df.drop(['listed_in(city)'],axis=1,inplace=True)
df.drop(['cuisines'],axis=1,inplace=True)

#print(len(df['location'].unique()))
#print(len(df['rest_type'].unique()))

a=pd.get_dummies(df['location'])
lc=[]
for i in a.columns:
    lc.append('location_'+i)
a.columns=lc
#print(a)
b=pd.get_dummies(df['rest_type'])
lc=[]
for i in b.columns:
    lc.append('rest_type_'+i)
b.columns=lc
#print(b)

df.drop(['location','rest_type'],axis=1,inplace=True)
df=pd.concat([df,a],axis=1)
df=pd.concat([df,b],axis=1)

a=pd.get_dummies(df['online_order'])
lc=[]
for i in a.columns:
    lc.append('online_order_'+i)
a.columns=lc
#print(a)
b=pd.get_dummies(df['book_table'])
lc=[]
for i in b.columns:
    lc.append('book_table_'+i)
b.columns=lc
#print(b)

df.drop(['online_order','book_table'],axis=1,inplace=True)
df=pd.concat([df,a],axis=1)
df=pd.concat([df,b],axis=1)

df['target']=df['rate'].apply(thresh)
df['rated']=df['rate'].apply(assign)

print(df.columns)
print(df.dtypes)

new_rest=df[df['rated']==0]
train_val_rest=df[df['rated']==1]
train_val_rest.drop(['rated'],axis=1,inplace=True)
train_val_rest = train_val_rest.sample(frac=1).reset_index(drop=True)

#print(train_val_rest.shape)

X=train_val_rest.drop(['target'],axis=1)
y=train_val_rest['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print(X_train.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

model = RandomForestClassifier()
#model = svm.SVC()
model.fit(X_train, y_train)

predicted = model.predict(X_test)
matrix = confusion_matrix(y_test, predicted)

print(matrix)
print(accuracy_score(y_test, predicted))
