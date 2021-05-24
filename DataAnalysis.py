import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import datetime
import calendar
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from folium.plugins import HeatMap
import folium

# LOAD AND MERGE ALL FILES INTO SINGLE FILE

path='D:/udemy_kurs_py/DataScienceProjects/Project7/GenerateData'
files=os.listdir(path)
csv_files=[]
for file in files:
    if file[-2:]=='py':
        continue
    elif file=='testdata.csv':
        continue
    else:
        csv_files.append(file)
#print(csv_files)

all_data=pd.DataFrame()

for csv_file in csv_files:
    df2=pd.read_csv(path+'/'+csv_file)
    all_data=pd.concat([df2,all_data])

all_data.to_csv('AllMonthlyData.csv', index=False)

# LOAD ALL DATA

df=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project7/AllMonthlyData.csv', parse_dates=['Order_Date'])

print(df.head())
print(df.tail())
print(df.isnull().sum())
print(df.dtypes)
print(df.describe())
print(df.shape)
print(df.columns)

# BUSY MONTH
 
df['Month']=df['Order_Date'].dt.month
df['TotalSpent']=df['Quantity_Ordered']*df['Price_Each']

monthly=df.groupby('Month')['TotalSpent'].sum().reset_index()
fig=px.bar(monthly, x='Month', y='TotalSpent', color='Month',
           title='Total Spent by Month',template='plotly_dark')
fig.show()

# BUSY CITY

def splitString(text):
    x=text.split(',')
    c=x[1].strip()
    s=x[2].strip().split(' ')[0]
    r=c+' '+s
    return r

df['City']=df['Purchase_Address'].apply(splitString)
#print(df.head())

city_wise=df.groupby('City')['TotalSpent'].sum().reset_index()
fig=px.bar(city_wise, x='City', y='TotalSpent', color='City',
           title='Total Spent by City',template='plotly_dark')
fig.show()

# BUSY HOUR

df['Hour']=df['Order_Date'].dt.hour

hourly=df.groupby('Hour')['OrderID'].count().reset_index()
fig=px.bar(hourly, x='Hour', y='OrderID', color='Hour',
           title='Busy hours',template='plotly_dark')
fig.show()

# PRODUCTS SOLD TOGETHER

duplikati=df[df.duplicated(subset='OrderID', keep=False)].reset_index(drop=True)
#print(duplikati.head())
#duplikati['JoinedProducts']=''
#idorders=duplikati['OrderID'].unique()

duplikati['JoinedProducts']=duplikati.groupby('OrderID')['Product'].transform(lambda x: ','.join(x))
print(duplikati.head())
duplikati2=duplikati[duplikati.duplicated(subset='OrderID', keep='first')].reset_index(drop=True)
print(duplikati2.head())

joinedProds=duplikati2.groupby('JoinedProducts')['OrderID'].count().reset_index()
joinedProds=joinedProds.sort_values(by='OrderID', ascending=False)
fig=px.bar(joinedProds.iloc[:10,:], x='JoinedProducts', y='OrderID', color='JoinedProducts',
           title='Products sold together',template='plotly_dark')
fig.show()

# MOST SOLD PRODUCT

prods=df.groupby('Product')['Quantity_Ordered'].sum().reset_index()
fig=px.bar(prods, x='Product', y='Quantity_Ordered', color='Product',
           title='Most sold product',template='plotly_dark')
fig.show()

df=df.sort_values(by='Price_Each', ascending=False)
listaprod=df.Product.unique()
listapric=[]
for i in listaprod:
    data12_=df[df['Product']==i]
    listapric.append(data12_['Price_Each'].unique()[0])
#prices=df.groupby('Product')['Price_Each'].mean()
listams=[]

for i in listaprod:
    data13_=prods[prods['Product']==i]
    listams.append(data13_['Quantity_Ordered'].unique()[0])

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Bar(x=listaprod, y=listams, name="Most sold products", marker=dict(color='green')),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=listaprod, y=listapric, name="Product Prices", mode='lines', marker=dict(color='red')),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Products sold/price",template='plotly_dark'
)

# Set x-axis title
fig.update_xaxes(title_text="Products")

# Set y-axes titles
fig.update_yaxes(title_text="Most sold products", secondary_y=False)
fig.update_yaxes(title_text="Product Prices", secondary_y=True)

fig.show()



