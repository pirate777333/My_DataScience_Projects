import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import plotly.express as px
from folium.plugins import HeatMap
import folium

# LOAD FILE

df=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project1/uber-pickups-in-new-york-city/Uber-Jan-Feb-FOIL.csv')
df['date']=pd.to_datetime(df['date'],format='%m/%d/%Y')
df['Month']=df['date'].dt.month
df['Day']=df['date'].dt.day
df['Weekday']=df['date'].dt.day_name()
print(df.tail())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

# ANALYSIS FOR BASE NUMBER

avb = df.groupby('dispatching_base_number')['active_vehicles'].sum().reset_index()

fig=px.bar(avb,x='dispatching_base_number',y='active_vehicles',color='dispatching_base_number',
           template='plotly_dark',title='Active vehicles per Base number')
fig.show()

trb = df.groupby('dispatching_base_number')['trips'].sum().reset_index()

fig=px.bar(trb,x='dispatching_base_number',y='trips',color='dispatching_base_number',
           template='plotly_dark',title='Trips per Base number')
fig.show()

df['tr/veh']=df['trips']/df['active_vehicles']

trvh = df.groupby('dispatching_base_number')['tr/veh'].sum().reset_index()

fig=px.bar(trvh,x='dispatching_base_number',y='tr/veh',color='dispatching_base_number',
           template='plotly_dark',title='Trips per Vehicles per Base number')
fig.show()
