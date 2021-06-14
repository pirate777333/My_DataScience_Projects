import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats import boxcox
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

def stringops(text):
    if text=='..':
        return None
    else:
        return text

# FOREST % LAND AREA

df=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Road/c3f97797-76e0-4565-b814-e508ec4e6244_Data.csv')
df=df.iloc[:-5,2:-1]
#print(df)
##print(df_fpla.head())
##print(df_fpla.tail(15))
##print(df_fpla.shape)
##print(df_fpla.columns)
##print(df_fpla.isnull().sum())
##print(df_fpla.dtypes)
##
stupci=['Country Name','Country Code']
for i in range(2000,2020):
    stupci.append(str(i))

df.columns=stupci

for i in df.columns:
    df[i]=df[i].apply(stringops)
    
df=df[['Country Name', 'Country Code', '2000',
       '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
       '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']]
#print(df.iloc[:-47,:])
countries=df.iloc[:-47,:]
world=df[df['Country Name']=='World']

list_of_codes=['ECS','MEA','EAS','SSF','LCN','NAC','SAS']

regions=df[df['Country Code'].isin(list_of_codes)]

#print(regions)
#print(countries.shape)
countries=countries.dropna()
#print(countries.shape)
for i in countries.columns[2:]:
    countries[i]=pd.to_numeric(countries[i])
for i in world.columns[2:]:
    world[i]=pd.to_numeric(world[i])
for i in regions.columns[2:]:
    regions[i]=pd.to_numeric(regions[i])

#print(regions.isnull().sum())
#print(world.isnull().sum())

croatia=countries[countries['Country Name']=='Croatia']

fig=px.line(x=croatia.columns[2:], y=croatia.iloc[:,2:].values[0],
            title="Road Mortality Rate per 100,000 people", template='plotly_dark')
fig.update_layout(xaxis_title='Years',
                  yaxis_title='Road Mortality Rate')
fig.show()

fig=px.line(x=world.columns[2:], y=world.iloc[:,2:].values[0],
            title="Road Mortality Rate per 100,000 people", template='plotly_dark')
fig.update_layout(xaxis_title='Years',
                  yaxis_title='Road Mortality Rate')
fig.show()

code_name={'ECS':'Europe and Central Asia',
           'MEA':'Middle East and North Africa',
           'EAS':'East Asia and Pacific',
           'SSF':'Sub-Saharan Africa',
           'LCN':'Latin America and Caribbean',
           'NAC':'North America',
           'SAS':'South Asia'}

boje=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']#, '#B6E880', '#FF97FF', '#FECB52']

fig = go.Figure()

b=0
for i in code_name.keys():
    regija=regions[regions['Country Code']==i]
    #print(code_name[i])
    #print(regija.columns[2:])
    #print(regija.iloc[:,2:].values[0])

    fig.add_trace(go.Scatter(x=regija.columns[2:], y=regija.iloc[:,2:].values[0],
                             name=code_name[i],mode='lines',
                             line=dict(color=boje[b], width=2)))
    b+=1

fig.update_layout(title='Road Mortality Rate per 100,000 people',
                   xaxis_title='Years',
                   yaxis_title='Road Mortality Rate',
                  template='plotly_dark')
fig.show()

regije_bar=[]
regije_bar_2018=[]
regije_bar_1990=[]

for i in code_name.keys():
    regija=regions[regions['Country Code']==i]
    regije_bar.append(code_name[i])
    regije_bar_2018.append(regija['2019'].values[0])
    regije_bar_1990.append(regija['2000'].values[0])

#print(regije_bar)
#print(regije_bar_2018)
#print(regije_bar_1990)

fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Bar(x=regije_bar, y=regije_bar_1990,name='2000'),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=regije_bar, y=regije_bar_2018,name='2019'),
    row=1, col=2
)

fig.update_layout(title_text="Road Mortality Rate per 100,000 people",
                  template='plotly_dark')
fig.show()

best2019=countries.sort_values(by='2019',ascending=False)
fig=px.bar(x=best2019.iloc[:10,0],y=best2019.iloc[:10,-1],color=best2019.iloc[:10,0],
           title='Highest road mortality rates 2019',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Road mortality rate')
fig.show()

best2019=countries.sort_values(by='2019',ascending=True)
fig=px.bar(x=best2019.iloc[:10,0],y=best2019.iloc[:10,-1],color=best2019.iloc[:10,0],
           title='Lowest road mortality rates 2019',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Road mortality rate')
fig.show()

best2000=countries.sort_values(by='2000',ascending=False)
fig=px.bar(x=best2000.iloc[:10,0],y=best2000.iloc[:10,2],color=best2000.iloc[:10,0],
           title='Highest road mortality rates 2000',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Road mortality rate')
fig.show()

countries['Difference']=countries['2019']-countries['2000']

bestch=countries.sort_values(by='Difference',ascending=False)
fig=px.bar(x=bestch.iloc[:10,0],y=bestch.iloc[:10,-1],color=bestch.iloc[:10,0],
           title='Road mortality rate uprise',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Road mortality rate (change 2019-2000)')
fig.show()

bestch=countries.sort_values(by='Difference',ascending=True)
fig=px.bar(x=bestch.iloc[:10,0],y=bestch.iloc[:10,-1],color=bestch.iloc[:10,0],
           title='Road mortality rate fall',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Road mortality rate (change 2019-2000)')
fig.show()

countries_px=countries[['Country Name','Country Code','2019']]
print(countries_px)

fig = go.Figure(data=go.Choropleth(
    locations=countries_px['Country Code'],
    z=countries_px['2019'].astype(float),
    colorscale='Reds',
    autocolorscale=False,
    text=countries_px['Country Name'], # hover text
    marker_line_color='red', # line markers between states
    colorbar_title="Road mortality rate per 100,000 people"
))

fig.update_layout(
    title_text='Road mortality rate',
    geo = dict(
        #projection=go.layout.geo.Projection(type = 'albers usa'),
        projection_type='equirectangular',
        showlakes=True, # lakes
        lakecolor='rgb(0, 0, 255)'),
)

fig.update_geos(
    resolution=50,
    showcoastlines=True, coastlinecolor="RebeccaPurple",
    #showland=True, landcolor="LightGreen",
    showocean=True, oceancolor="LightBlue",
    #showlakes=True, lakecolor="Blue",
    showrivers=True, rivercolor="Blue"
)

fig.show()

