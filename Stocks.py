import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from folium.plugins import HeatMap
import folium

# LOAD AND SAVE TO ALL DATA

apple=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project5/individual_stocks_5yr/AAPL_data.csv')
google=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project5/individual_stocks_5yr/GOOG_data.csv')
microsoft=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project5/individual_stocks_5yr/MSFT_data.csv')
amazon=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project5/individual_stocks_5yr/AMZN_data.csv')

all_data=pd.DataFrame()
all_data=pd.concat([apple, all_data])
all_data=pd.concat([google, all_data])
all_data=pd.concat([microsoft, all_data])
all_data=pd.concat([amazon, all_data])

all_data.to_csv('Stocks_All4Companies.csv', index=False)

# LOAD ALL DATA

df=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project5/Stocks_All4Companies.csv', parse_dates=['date'])

print(df.head())
print(df.tail())
print(df.isnull().sum())
print(df.dtypes)
print(df.describe())
print(df.shape)
print(df.columns)

# ANALYSIS BY CLOSE 

names=df.Name.unique()
print(names)

fig = go.Figure()

for name in names:
    df_name=df[df['Name']==name]

    fig.add_trace(go.Scatter(
        x=df_name['date'],
        y=df_name['close'],
        name = name,
        mode='lines'
    ))    

fig.update_layout(template='plotly_dark', title='Closing prices')
fig.show()

# ANALYSIS BY VOLUME

fig = go.Figure()

for name in names:
    dfv_name=df[df['Name']==name]

    fig.add_trace(go.Scatter(
        x=dfv_name['date'],
        y=dfv_name['volume'],
        name = name,
        mode='lines'
    ))    

fig.update_layout(template='plotly_dark', title='Volume prices')
fig.show()

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(names[0], names[1], names[2], names[3]))

dfv_name=df[df['Name']==names[0]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['volume'],
                         mode='lines',
                         name=names[0]),
              row=1, col=1)

dfv_name=df[df['Name']==names[1]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['volume'],
                         mode='lines',
                         name=names[1]),
              row=1, col=2)

dfv_name=df[df['Name']==names[2]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['volume'],
                         mode='lines',
                         name=names[2]),
              row=2, col=1)

dfv_name=df[df['Name']==names[3]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['volume'],
                         mode='lines',
                         name=names[3]),
              row=2, col=2)

fig.update_layout(template='plotly_dark',
                  title_text="Volume")

fig.show()

# DAILY PRICE CHANGE

df['daily_change']=df['close']-df['open']

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(names[0], names[1], names[2], names[3]))

dfv_name=df[df['Name']==names[0]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['daily_change'],
                         mode='lines',
                         name=names[0]),
              row=1, col=1)

dfv_name=df[df['Name']==names[1]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['daily_change'],
                         mode='lines',
                         name=names[1]),
              row=1, col=2)

dfv_name=df[df['Name']==names[2]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['daily_change'],
                         mode='lines',
                         name=names[2]),
              row=2, col=1)

dfv_name=df[df['Name']==names[3]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['daily_change'],
                         mode='lines',
                         name=names[3]),
              row=2, col=2)

fig.update_layout(template='plotly_dark',
                  title_text="Daily change of prices")

fig.show()

df['daily_perc_return']=(df['close']-df['open'])/df['close']*100

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(names[0], names[1], names[2], names[3]))

dfv_name=df[df['Name']==names[0]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['daily_perc_return'],
                         mode='lines',
                         name=names[0]),
              row=1, col=1)

dfv_name=df[df['Name']==names[1]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['daily_perc_return'],
                         mode='lines',
                         name=names[1]),
              row=1, col=2)

dfv_name=df[df['Name']==names[2]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['daily_perc_return'],
                         mode='lines',
                         name=names[2]),
              row=2, col=1)

dfv_name=df[df['Name']==names[3]]

fig.add_trace(go.Scatter(x=dfv_name['date'],
                         y=dfv_name['daily_perc_return'],
                         mode='lines',
                         name=names[3]),
              row=2, col=2)

fig.update_layout(template='plotly_dark',
                  title_text="Daily change of prices percentage")

fig.show()

# MONTHLY MEAN OF CLOSE
df['Year']=df['date'].dt.year
df['Month']=df['date'].dt.month
#print(df['Year'].unique())
#print(df['Month'].unique())

for name in names:
    df2=df[df['Name']==name]
    df3=df2.groupby('Month')['close'].mean().reset_index()
    fig=px.bar(df3, x='Month',y='close', color='Month',
               template='plotly_dark',title=name+' Mean of close prices')
    fig.show()

for name in names:
    df4=df[df['Name']==name]
    df5=df4.groupby(['Year','Month'])['close'].mean().reset_index()
    df5['Year']=df5['Year'].astype(str)
    df5['Month']=df5['Month'].astype(str)
    #print(df5.dtypes)
    df5['datum']=df5['Year']+'-'+df5['Month']
    df5['datum']=pd.to_datetime(df5['datum'],format='%Y-%m')
    fig=px.line(df5, x='datum',y='close', #color='Year',
               template='plotly_dark',title=name+' Mean of close prices')
    fig.show()
names=['AMZN', 'MSFT', 'AAPL']
for name in names:
    df6=df[df['Name']==name]
    df7=df6.groupby(['Year','Month'])['close'].mean().reset_index()
    godine=df7.Year.unique()
    print(godine)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(str(godine[0]), str(godine[1]), str(godine[2]), str(godine[3]), str(godine[4]), str(godine[5])))

    dfgod=df7[df7['Year']==godine[0]]

    fig.add_trace(go.Bar(x=dfgod['Month'],
                             y=dfgod['close'],
                             name='Year'+str(godine[0])),
                  row=1, col=1)

    dfgod=df7[df7['Year']==godine[1]]

    fig.add_trace(go.Bar(x=dfgod['Month'],
                             y=dfgod['close'],
                             name=str(godine[1])),
                  row=1, col=2)

    dfgod=df7[df7['Year']==godine[2]]

    fig.add_trace(go.Bar(x=dfgod['Month'],
                             y=dfgod['close'],
                             name=str(godine[2])),
                  row=1, col=3)

    dfgod=df7[df7['Year']==godine[3]]

    fig.add_trace(go.Bar(x=dfgod['Month'],
                             y=dfgod['close'],
                             name=str(godine[3])),
                  row=2, col=1)


    dfgod=df7[df7['Year']==godine[4]]

    fig.add_trace(go.Bar(x=dfgod['Month'],
                             y=dfgod['close'],
                             name=str(godine[4])),
                  row=2, col=2)

    dfgod=df7[df7['Year']==godine[5]]

    fig.add_trace(go.Bar(x=dfgod['Month'],
                             y=dfgod['close'],
                             name=str(godine[5])),
                  row=2, col=3)

    fig.update_layout(template='plotly_dark',
                      title_text="Monthly and Yearly mean of Close prices for "+name)

    fig.show()

# ARE COMPANIES CORRELATED?

names=df.Name.unique()

closing=pd.DataFrame()

for i in names:
    df20=df[df.Name==i]
    closing=pd.concat([df20.close.to_frame().reset_index(drop=True),closing],axis=1,ignore_index=True)

closing.columns=names
print(closing)

sns.heatmap(closing.corr(),annot=True)
bottom, top = plt.ylim()
plt.ylim(bottom+0.5, top-0.5)
plt.show()

dailydf=pd.DataFrame()

for i in names:
    df30=df[df.Name==i]
    df30['daily_return']=df30['close']-df30['open']
    dailydf=pd.concat([df30.daily_return.to_frame().reset_index(drop=True),dailydf],axis=1,ignore_index=True)

dailydf.columns=names
print(dailydf)

sns.heatmap(dailydf.corr(),annot=True)
bottom, top = plt.ylim()
plt.ylim(bottom+0.5, top-0.5)
plt.show()

dailydf=pd.DataFrame()

for i in names:
    df40=df[df.Name==i]
    df40['daily_return']=(df40['close']-df40['open'])/df40['close']*100
    dailydf=pd.concat([df40.daily_return.to_frame().reset_index(drop=True),dailydf],axis=1,ignore_index=True)

dailydf.columns=names
print(dailydf)

sns.heatmap(dailydf.corr(),annot=True)
bottom, top = plt.ylim()
plt.ylim(bottom+0.5, top-0.5)
plt.show()
