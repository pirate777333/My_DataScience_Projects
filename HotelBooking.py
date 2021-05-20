import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import HeatMap
import folium

# LOAD FILE

df=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project2/hotel_bookings.csv')
##df['date']=pd.to_datetime(df['date'],format='%m/%d/%Y')
##df['Month']=df['date'].dt.month
##df['Day']=df['date'].dt.day
##df['Weekday']=df['date'].dt.day_name()
df.fillna(0,inplace=True)
df=df[(df['babies']!=0)&(df['adults']!=0)&(df['babies']!=0)]
all_data=df.copy()
all_cancel=df[df['is_canceled']==1]
df=df[df['is_canceled']==0]
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

print(df['market_segment'])

# ANALYSIS FOR COUNTRY

c = df.groupby('country')['hotel'].count().reset_index()
c=c[c['country']!=0]
print(c)
fig=px.bar(c,x='country',y='hotel',color='country',
           template='plotly_dark',title='Countries')
fig.show()

fig = px.pie(c, values='hotel', names='country', title='Countries')
fig.show()

fig=px.choropleth(c,
                  locations=c['country'],
                  color=c['hotel'],
                  hover_name=c['country'])
fig.show()

# PRICES PER ROOM

lista=df.reserved_room_type.unique()

for i in lista:
    df2=df[df['reserved_room_type']==i]
    fig = px.box(df2, y='adr',#color='Weekday',
             template='plotly_dark',title='Prices per room type: '+str(i))
    fig.show()

# PRICES PER YEAR

lista=df.arrival_date_year.unique()

for i in lista:
    df2=df[df['arrival_date_year']==i]
    df3=df2.groupby('arrival_date_month')['adr'].mean().reset_index()
    fig = px.line(df3, x='arrival_date_month', y='adr',#color='Weekday',
             template='plotly_dark',title='Prices per room: '+str(i))
    fig.show()

# NIGHTS SPENT AT HOTEL / MARKET_SEGMENT
resort=df[df['hotel']=='Resort Hotel']
city=df[df['hotel']=='City Hotel']

sns.boxplot(x='market_segment',y='stays_in_weekend_nights', data=resort)
plt.show()
sns.boxplot(x='market_segment',y='stays_in_weekend_nights', data=city)
plt.show()
sns.boxplot(x='market_segment',y='stays_in_weekend_nights', data=df,hue='hotel')
plt.show()

# PREFERENCES

meals=df.groupby('meal')['hotel'].count().reset_index()
fig=px.bar(meals,x='meal',y='hotel',color='meal',
           template='plotly_dark',title='Countries')
fig.show()

# RELATIONSHIP SPEC REQ / CANCEL

relationship=all_data.groupby(['total_of_special_requests','is_canceled'])['hotel'].count().reset_index()
print(relationship)

fig=px.bar(relationship,x='total_of_special_requests',y='hotel',color='is_canceled',
           template='plotly_dark',title='Relationship between requests and cancelations')
fig.show()

# BUSY MONTHS

monthly = df.groupby('arrival_date_month')['hotel'].count().reset_index()
fig=px.bar(monthly,x='arrival_date_month',y='hotel',color='arrival_date_month',
           template='plotly_dark',title='Busy Months')
fig.show()

# HOW LONG PEOPLE STAY

resort=df[df['hotel']=='Resort Hotel']
city=df[df['hotel']=='City Hotel']

fig = go.Figure()
fig.add_trace(go.Box(y=resort['stays_in_weekend_nights']))
fig.add_trace(go.Box(y=resort['stays_in_week_nights']))
fig.add_trace(go.Box(y=city['stays_in_weekend_nights']))
fig.add_trace(go.Box(y=city['stays_in_week_nights']))
fig.show()

# BOOKING BY MARKET SEGMENT

markseg=df.groupby(['market_segment','hotel'])['lead_time'].count().reset_index()
fig=px.bar(markseg,x='market_segment',y='lead_time',color='hotel',
           template='plotly_dark',title='Market Segments')
fig.show()

# CANCELLED

print(all_cancel.shape[0],' are cancelled')

nogo=all_cancel.groupby(['arrival_date_month','hotel'])['lead_time'].count().reset_index()
fig=px.bar(nogo,x='arrival_date_month',y='lead_time',color='hotel',
           template='plotly_dark',title='Cancelled')
fig.show()

