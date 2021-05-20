import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import plotly.express as px
from folium.plugins import HeatMap
import folium

# COLLECT ALL FILES

path='D:/udemy_kurs_py/DataScienceProjects/Project1/uber-pickups-in-new-york-city'
files=os.listdir(path)[-7:]
files.remove('uber-raw-data-janjune-15.csv')
print(files)

all_data = pd.DataFrame()

for file in files:
    df=pd.read_csv(path+'/'+file)
    all_data=pd.concat([df, all_data])

print(all_data.head())
print(all_data.shape)
all_data.to_csv('all_data_UberNYC.csv',index=False)

# READ NEW FILE; PREPARATION & PREPROCESSING

newfilepath='all_data_UberNYC.csv'
df=pd.read_csv(newfilepath)
df=df.iloc[:,1:]
print(df.head())
print(df.isnull().sum())
print(df.dtypes)
df['Date/Time']=pd.to_datetime(df['Date/Time'],format='%m/%d/%Y %H:%M:%S')
print(df.dtypes)
df['Weekday']=df['Date/Time'].dt.day_name()
df['Day']=df['Date/Time'].dt.day
df['Month']=df['Date/Time'].dt.month
df['Year']=df['Date/Time'].dt.year
df['Minute']=df['Date/Time'].dt.minute
df['Hour']=df['Date/Time'].dt.hour
print(df.head())
df.to_csv('all_data_Prep_UberNYC.csv',index=False)

# READ NEW FILE; ANALYSIS

newfilepath2='all_data_Prep_UberNYC.csv'
df_new=pd.read_csv(newfilepath2,parse_dates=['Date/Time'])
print(df_new.head())
print(df_new.shape)
print(df_new.dtypes)

# ANALYSIS BY WEEKDAYS

weekdays=df_new.groupby('Weekday')['Date/Time'].count().reset_index()
fig = px.bar(weekdays, x='Weekday', y='Date/Time',color='Weekday',
             template='plotly_dark',title='Analysis of Uber trips by Weekdays')
fig.show()

# ANALYSIS BY Hours

hours=df_new.groupby('Hour')['Date/Time'].count().reset_index()
fig = px.bar(hours, x='Hour', y='Date/Time',#color='Weekday',
             template='plotly_dark',title='Analysis of Uber trips by Hours')
fig.show()

hours_monthly=df_new.groupby(['Month','Hour'])['Date/Time'].count().reset_index()
lista=hours_monthly.Month.unique()

for i in lista:
    df=hours_monthly[hours_monthly['Month']==i]
    fig = px.bar(df, x='Hour', y='Date/Time',#color='Weekday',
             template='plotly_dark',title='Analysis of Uber trips by Hours for Month: '+str(i))
    fig.show()

# ANALYSIS BY MONTHS

months=df_new.groupby('Month')['Date/Time'].count().reset_index()
fig = px.bar(months, x='Month', y='Date/Time',color='Month',
             template='plotly_dark',title='Analysis of Uber trips by Months')
fig.show()

# ANALYSIS BY DAYS

days=df_new.groupby('Day')['Date/Time'].count().reset_index()
fig = px.bar(days, x='Day', y='Date/Time',color='Day',
             template='plotly_dark',title='Analysis of Uber trips by Days')
fig.show()

# ANALYSIS BY EACH MONTH / DAILY

days_monthly=df_new.groupby(['Month','Day'])['Date/Time'].count().reset_index()
lista=days_monthly.Month.unique()

for i in lista:
    df=days_monthly[days_monthly['Month']==i]
    fig = px.bar(df, x='Day', y='Date/Time',#color='Weekday',
             template='plotly_dark',title='Analysis of Uber trips by Days for Month: '+str(i))
    fig.show()

# ANALYSIS BY EACH WEEKDAY / HOURLY

hours_weekday=df_new.groupby(['Weekday','Hour'])['Date/Time'].count().reset_index()
lista=hours_weekday.Weekday.unique()

for i in lista:
    df=hours_weekday[hours_weekday['Weekday']==i]
    fig = px.bar(df, x='Hour', y='Date/Time',#color='Weekday',
             template='plotly_dark',title='Analysis of Uber trips by Hours for: '+str(i))
    fig.show()

# ANALYSIS BY BASE_NAME / MONTHLY

base_monthly=df_new.groupby(['Base','Month'])['Date/Time'].count().reset_index()
lista=base_monthly.Base.unique()

for i in lista:
    df=base_monthly[base_monthly['Base']==i]
    fig = px.bar(df, x='Month', y='Date/Time',#color='Weekday',
             template='plotly_dark',title='Analysis of Uber trips by Base for: '+str(i))
    fig.show()

# HEATMAPS

weekday_hour_hm=df_new.groupby(['Weekday','Hour'])['Date/Time'].count().unstack()
print(weekday_hour_hm)
sns.heatmap(weekday_hour_hm)
plt.show()

heatmap_f=df_new.groupby(['Lat','Lon'])['Date/Time'].count().reset_index()

heatmap_f.columns=['Lat','Lon','Count']
heatmap_f=heatmap_f[(heatmap_f['Lat']!=0)&(heatmap_f['Lon']!=0)]

basemap=folium.Map()
HeatMap(heatmap_f,zoom=20,radius=15).add_to(basemap)
basemap.save('daily.html')
