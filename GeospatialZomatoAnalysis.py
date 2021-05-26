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

def splitString(text):
    x=text.split('/')[0].strip()
    return x

def replaceString(text):
    x=text.replace(',','')
    return x

# LOAD N EXPLORE

file='D:/udemy_kurs_py/DataScienceProjects/Project8/zomato.csv'

df=pd.read_csv(file)

print(df.head())
print(df.tail())
print(df.isnull().sum())
print(df.dtypes)
print(df.describe())
print(df.shape)
print(df.columns)

df.dropna(axis='index', subset=['rate'],inplace=True)

df['rating_out5']=df['rate'].apply(splitString)

df.replace('NEW',0,inplace=True)
df.replace('-',0,inplace=True)

df['rating_out5']=pd.to_numeric(df['rating_out5'])

#print(df.head())

# AVG RATE OF RESTAURANTS

avgrat=df.groupby('name')['rating_out5'].mean().reset_index()
avgrat=avgrat.sort_values(by='rating_out5', ascending=False)
fig=px.bar(avgrat.iloc[:10,:], x='name', y='rating_out5', color='name',
           title='Best rated Restaurants',template='plotly_dark')
fig.show()

avgratmin=avgrat.sort_values(by='rating_out5', ascending=True)
fig=px.bar(avgratmin.iloc[1500:1510,:], x='name', y='rating_out5', color='name',
           title='Best rated Restaurants',template='plotly_dark')
fig.show()

print(df['rating_out5'].describe())

chains=df.groupby('name')['rating_out5'].count().reset_index()
chains=chains.sort_values(by='rating_out5', ascending=False)
fig=px.bar(chains.iloc[:10,:], x='name', y='rating_out5', color='name',
           title='Chains',template='plotly_dark')
fig.show()

# ONLINE ORDER

print(df['online_order'].unique())

onlineOrd=df.groupby('online_order')['name'].count().reset_index()
fig=px.pie(onlineOrd, values='name', names='online_order',
           title='Online Orders ?', template='plotly_dark')
fig.show()

# BOOK TABLE ?

print(df['book_table'].unique())

bookTable=df.groupby('book_table')['name'].count().reset_index()
fig=px.pie(bookTable, values='name', names='book_table',
           title='Book Table ?', template='plotly_dark')
fig.show()

# RESTAURANTS TYPES

print(df['rest_type'].unique())

restTypes=df.groupby('rest_type')['name'].count().reset_index()
fig=px.pie(restTypes.iloc[:10,:], values='name', names='rest_type',
           title='Restaurant Types', template='plotly_dark')
fig.show()

# HIGHEST VOTED

print(df['votes'].unique())

glasanje=df.groupby('name')['votes'].mean().reset_index()
glasanje=glasanje.sort_values(by='votes', ascending=False)
fig=px.bar(glasanje.iloc[:10,:], x='name', y='votes', color='name',
           title='Highest Voted Restaurants',template='plotly_dark')
fig.show()

# LOCATION OF RESTAURANTS

print(df['location'].unique())

locations=df.groupby('location')['name'].count().reset_index()
fig=px.pie(locations.iloc[:10,:], values='name', names='location',
           title='Restaurant Locations', template='plotly_dark')
fig.show()

# WHICH COUSINES?

kuzine=df.groupby('cuisines')['name'].count().reset_index()
fig=px.pie(kuzine.iloc[:10,:], values='name', names='cuisines',
           title='Restaurant Cuisines', template='plotly_dark')
fig.show()

# APPROX 2 PEOPLE COST

df_2people=df.dropna(axis='index', subset=['approx_cost(for two people)'])
df_2people['approx_cost(for two people)']=df_2people['approx_cost(for two people)'].apply(replaceString)
df_2people['approx_cost(for two people)']=pd.to_numeric(df_2people['approx_cost(for two people)'])
print(df_2people['approx_cost(for two people)'])

twopeople=df_2people.groupby('name')['approx_cost(for two people)'].mean().reset_index()
twopeople=twopeople.sort_values(by='approx_cost(for two people)', ascending=False)
fig=px.bar(twopeople.iloc[:10,:], x='name', y='approx_cost(for two people)', color='name',
           title='Highest Priced Restaurants for 2 people',template='plotly_dark')
fig.show()

# RATING/PRICE

vsdf1=pd.DataFrame(columns=['Rating','2PeopleCost'])
vsdf1['Rating']=avgrat['rating_out5'].values[:7135]
vsdf1['2PeopleCost']=twopeople['approx_cost(for two people)'].values[:7135]
print(vsdf1)

fig=px.scatter(x=df['rating_out5'],y=df['approx_cost(for two people)'], color=df['online_order'],
               template='plotly_dark', title='RATING/PRICE')
fig.show()
fig=px.scatter(x=df['votes'],y=df['approx_cost(for two people)'], color=df['online_order'],
               template='plotly_dark', title='RATING/PRICE')
fig.show()
fig=px.scatter(x=df['rating_out5'],y=df['approx_cost(for two people)'], color=df['book_table'],
               template='plotly_dark', title='RATING/PRICE')
fig.show()

# Top 10 Luxuriest

luxury=df_2people.sort_values(by='approx_cost(for two people)', ascending=False)
fig=px.bar(luxury.iloc[:10,:], x='name', y='approx_cost(for two people)', color='name',
           title='Highest Priced Restaurants for 2 people',template='plotly_dark')
fig.show()

# DISTRIBUTION

sns.distplot(luxury['approx_cost(for two people)'])
plt.show()

twopeople2=df_2people.groupby('name')['approx_cost(for two people)'].mean().reset_index()
twopeople2=twopeople2.sort_values(by='approx_cost(for two people)', ascending=True)
fig=px.bar(twopeople2.iloc[:10,:], x='name', y='approx_cost(for two people)', color='name',
           title='Highest Priced Restaurants for 2 people',template='plotly_dark')
fig.show()

# NAJBOLJI OMJER CIJENE I KVALITETE XIAOMI

df_2people_2=df_2people[(df_2people['rating_out5']>4.0)&(df_2people['approx_cost(for two people)']<500)]
df_2people_2=df_2people_2.drop_duplicates(subset=['name'])
fig=px.bar(df_2people_2, x='name', y='approx_cost(for two people)', color='rating_out5',
           title='NAJBOLJI OMJER CIJENE I KVALITETE',template='plotly_dark')
fig.show()
#print(df_2people_2)

df_2people_3=df_2people.groupby('location')['name'].min('approx_cost(for two people)').reset_index()

print(df_2people_3)

df_2people_4=df_2people.groupby(['location','name']).min('approx_cost(for two people)').reset_index()
df_2people_42=df_2people_4.groupby('location')['name'].count().reset_index()
df_2people_42=df_2people_42.sort_values(by='name',ascending=False)
fig=px.bar(df_2people_42.iloc[:10,:], x='location', y='name', color='location',
           title='Best Locations',template='plotly_dark')
fig.show()

# LOCATIONS HEATMAP

lokacije_=pd.DataFrame({'Name':df_2people['location'].unique()})

from geopy.geocoders import Nominatim
geolocator=Nominatim(user_agent='app')

lat_lon=[]
for location in lokacije_['Name']:
    lok=geolocator.geocode(location)
    if lok is None:
        lat_lon.append(np.nan)
    else:
        geo=(lok.latitude,lok.longitude)
        lat_lon.append(geo)

lokacije_['geo_loc']=lat_lon


rest_locations=pd.DataFrame(df_2people['location'].value_counts().reset_index())
rest_locations.columns=['Name','Count']

RestoraniLokacije=rest_locations.merge(lokacije_,on='Name', how='left').dropna()

lat,lon=zip(*np.array(RestoraniLokacije['geo_loc']))

RestoraniLokacije['lat']=lat
RestoraniLokacije['lon']=lon

RestoraniLokacije.drop('geo_loc',axis=1,inplace=True)

def generatebasemap(default_location=[12.97,77.59],default_zoom_start=12):
    basemap=folium.Map(location=default_location,zoom_start=default_zoom_start)
    return basemap

basemap=generatebasemap()

HeatMap(RestoraniLokacije[['lat','lon','Count']].values.tolist(),zoom=20,radius=15).add_to(basemap)
basemap.save('restloc.html')

df_novi=df_2people[df_2people['cuisines']=='North Indian']

ni=df_novi.groupby(['location'],as_index=False)['url'].agg('count')
ni.columns=['Name','Count']

RestoraniLokacijeN=ni.merge(lokacije_,on='Name', how='left').dropna()

lat,lon=zip(*np.array(RestoraniLokacijeN['geo_loc']))

RestoraniLokacijeN['lat']=lat
RestoraniLokacijeN['lon']=lon

RestoraniLokacijeN.drop('geo_loc',axis=1,inplace=True)

def generatebasemap(default_location=[12.97,77.59],default_zoom_start=12):
    basemap=folium.Map(location=default_location,zoom_start=default_zoom_start)
    return basemap

basemap=generatebasemap()

HeatMap(RestoraniLokacijeN[['lat','lon','Count']].values.tolist(),zoom=20,radius=15).add_to(basemap)
basemap.save('restloc_north.html')
