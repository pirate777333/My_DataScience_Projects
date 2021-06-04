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

file='D:/udemy_kurs_py/DataScienceProjects/Project95/Pandas_Project_Session/data/global_superstore/global_superstore_20162.csv'

import chardet
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

def replacestring(text):
    chars=['$','(',')']
    for c in chars:
        if c in text:
            text=text.replace(c,'')
    return text

df = pd.read_csv(file,encoding='Windows-1252', parse_dates=['Order Date','Ship Date'])

df.dropna(inplace=True)

df['Sales']=df['Sales'].apply(replacestring)
df['Profit']=df['Profit'].apply(replacestring)

df['Sales']=pd.to_numeric(df['Sales'])
df['Profit']=pd.to_numeric(df['Profit'])

print(df.head())
print(df.columns)
print(df.dtypes)
print(df.tail())
print(df.shape)
print(df.describe())
print(df.isnull().sum())

# SHIPPING OVER 500

sh500=df[df['Shipping Cost']>300]
print(sh500.shape[0])

# COUNTRIES

print(df.State.unique())
print(len(df.Region.unique()))

# MULTIPLE CONDITIONS

multi=df[(df['Shipping Cost']>300)&(df['Region']=='Central US')]
print(multi.shape[0])
print(multi['Order ID'])

multi2=df[(df['State']=='Kentucky')&(df['Category']=='Technology')]
print(multi2.shape[0])

multi3=df[(df['Profit']>=500)&(df['Sales']>=300)]
print(multi3['State'].unique())

# MAX PROFIT

tech = df[df['Category']=='Technology']
print(tech['State'].unique())

techgr=tech.groupby('State')['Profit'].max().reset_index()
print(techgr)

profitmax=df.groupby(['Category','Sub-Category'])['Profit'].max().reset_index()
profitmin=df.groupby(['Category','Sub-Category'])['Profit'].min().reset_index()

print(profitmax)
print(profitmin)

salemax=df.groupby(['Category','Sub-Category'])['Sales'].max().reset_index()
salemin=df.groupby(['Category','Sub-Category'])['Sales'].min().reset_index()

print(salemax)
print(salemin)

sns.countplot(df['Ship Mode'])
plt.show()

cities=df.groupby('City')['Order ID'].count().reset_index()
cities=cities.sort_values('Order ID', ascending=False)
fig=px.bar(cities.iloc[:5,:],x='City',y='Order ID',title='Cities',template='plotly_dark')
fig.show()

cat=df.groupby('Category')['Order ID'].count().reset_index()
cat=cat.sort_values('Order ID', ascending=False)
fig=px.bar(cat.iloc[:5,:],x='Category',y='Order ID',title='Category',template='plotly_dark')
fig.show()

subcat=df.groupby('Sub-Category')['Order ID'].count().reset_index()
subcat=subcat.sort_values('Order ID', ascending=False)
fig=px.bar(subcat.iloc[:5,:],x='Sub-Category',y='Order ID',title='Sub-Category',template='plotly_dark')
fig.show()

cat=df.groupby('Category')['Profit'].sum().reset_index()
cat=cat.sort_values('Profit', ascending=False)
fig=px.bar(cat.iloc[:5,:],x='Category',y='Profit',title='Category',template='plotly_dark')
fig.show()

cat=df.groupby('Category')['Sales'].sum().reset_index()
cat=cat.sort_values('Sales', ascending=False)
fig=px.bar(cat.iloc[:5,:],x='Category',y='Sales',title='Category',template='plotly_dark')
fig.show()

cat=df.groupby('Sub-Category')['Profit'].sum().reset_index()
cat=cat.sort_values('Profit', ascending=False)
fig=px.bar(cat.iloc[:5,:],x='Sub-Category',y='Profit',title='Sub-Category',template='plotly_dark')
fig.show()

cat=df.groupby('Sub-Category')['Sales'].sum().reset_index()
cat=cat.sort_values('Sales', ascending=False)
fig=px.bar(cat.iloc[:5,:],x='Sub-Category',y='Sales',title='Sub-Category',template='plotly_dark')
fig.show()
