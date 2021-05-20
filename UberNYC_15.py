import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import plotly.express as px
from folium.plugins import HeatMap
import folium

# LOAD FILE

df=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project1/uber-pickups-in-new-york-city/uber-raw-data-janjune-15.csv')
df['Pickup_date']=pd.to_datetime(df['Pickup_date'],format='%Y-%m-%d %H:%M:%S')
df['Month']=df['Pickup_date'].dt.month
df['Day']=df['Pickup_date'].dt.day
df['Weekday']=df['Pickup_date'].dt.day_name()
df['Hour']=df['Pickup_date'].dt.hour
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

# ANALYSIS BY MONTH

monthly=df.groupby('Month')['Pickup_date'].count().reset_index()
fig=px.bar(monthly,x='Month',y='Pickup_date',color='Month',
           template='plotly_dark',title='Analysis by Month')
fig.show()

# ANALYSIS BY HOUR

hourly=df.groupby('Hour')['Pickup_date'].count().reset_index()
fig=px.bar(hourly,x='Hour',y='Pickup_date',#color='Month',
           template='plotly_dark',title='Analysis by Hour')
fig.show()

hour_weekday=df.groupby(['Weekday','Hour'])['Pickup_date'].count().reset_index()
lista=hour_weekday.Weekday.unique()

for i in lista:
    df2=hour_weekday[hour_weekday['Weekday']==i]
    fig = px.bar(df2, x='Hour', y='Pickup_date',#color='Weekday',
             template='plotly_dark',title='Analysis of Uber trips by Weekday for: '+str(i))
    fig.show()

# ANALYSIS BY DAY

daily=df.groupby('Day')['Pickup_date'].count().reset_index()
fig=px.bar(daily,x='Day',y='Pickup_date',#color='Month',
           template='plotly_dark',title='Analysis by Day')
fig.show()

day_weekday=df.groupby(['Month','Weekday'])['Pickup_date'].count().reset_index()
lista=day_weekday.Month.unique()

for i in lista:
    df3=day_weekday[day_weekday['Month']==i]
    fig = px.bar(df3, x='Weekday', y='Pickup_date',#color='Weekday',
             template='plotly_dark',title='Analysis of Uber trips by Month for: '+str(i))
    fig.show()

