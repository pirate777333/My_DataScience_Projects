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

# FOREST SQUARE METER

df_fsqm=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/ProjectForests/87e9a8e3-d80c-4325-941b-1a4f70b8a945_Data.csv')
df_fsqm=df_fsqm.iloc[:-5,2:-2]
#print(df_fsqm.head())
#print(df_fsqm.tail(15))
#print(df_fsqm.shape)
#print(df_fsqm.columns)
#print(df_fsqm.isnull().sum())
#print(df_fsqm.dtypes)

stupci=['Country Name','Country Code']
for i in range(1960,2019):
    stupci.append(str(i))

df_fsqm.columns=stupci

for i in df_fsqm.columns:
    df_fsqm[i]=df_fsqm[i].apply(stringops)
    
df_fsqm=df_fsqm[['Country Name', 'Country Code','1990', '1991',
       '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000',
       '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
       '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']]

countries=df_fsqm.iloc[:-8,:]
world=df_fsqm[df_fsqm['Country Name']=='World']
world=world.iloc[:,:-2]
regions=df_fsqm.iloc[-7:,:]
#print(regions)

#print(countries.shape)

countries=countries.dropna()
for i in countries.columns[2:]:
    countries[i]=pd.to_numeric(countries[i])
for i in world.columns[2:]:
    world[i]=pd.to_numeric(world[i])
for i in regions.columns[2:]:
    regions[i]=pd.to_numeric(regions[i])
    
#print(countries.shape)

croatia=countries[countries['Country Name']=='Croatia']

fig=px.line(x=croatia.columns[2:], y=croatia.iloc[:,2:].values[0],
            title="Croatia Forest Area in sq.m.", template='plotly_dark')
fig.update_layout(xaxis_title='Years',
                  yaxis_title='Croatia Land Area in sq.m.')
fig.show()


best2018=countries.sort_values(by='2018',ascending=False)
fig=px.bar(x=best2018.iloc[:10,0],y=best2018.iloc[:10,-1],color=best2018.iloc[:10,0],
           title='Best 10 in 2018',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Forest Land Area in sq. m.')
fig.show()

best1990=countries.sort_values(by='1990',ascending=False)
fig=px.bar(x=best1990.iloc[:10,0],y=best1990.iloc[:10,2],color=best1990.iloc[:10,0],
           title='Best 10 in 1990',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Forest Land Area in sq. m.')
fig.show()

countries['Difference']=countries['2018']-countries['1990']
countries['Difference_perc']=(countries['2018']-countries['1990'])/countries['1990']

#print(countries.head())

bestch=countries.sort_values(by='Difference',ascending=False)
fig=px.bar(x=bestch.iloc[:10,0],y=bestch.iloc[:10,-2],color=bestch.iloc[:10,0],
           title='Best 10 in positive change',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Forest Land Area in sq. m. (change 2018-1990)')
fig.show()

bestchperc=countries.sort_values(by='Difference_perc',ascending=False)
fig=px.bar(x=bestchperc.iloc[:10,0],y=bestchperc.iloc[:10,-1],color=bestchperc.iloc[:10,0],
           title='Best 10 in positive change (%)',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Forest Land Area in % (change 2018-1990)')
fig.show()

bestch=countries.sort_values(by='Difference',ascending=True)
fig=px.bar(x=bestch.iloc[:10,0],y=bestch.iloc[:10,-2],color=bestch.iloc[:10,0],
           title='Worst 10 in change',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Forest Land Area in sq. m. (change 2018-1990)')
fig.show()

bestchperc=countries.sort_values(by='Difference_perc',ascending=True)
fig=px.bar(x=bestchperc.iloc[:10,0],y=bestchperc.iloc[:10,-1],color=bestchperc.iloc[:10,0],
           title='Worst 10 in change (%)',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Forest Land Area in % (change 2018-1990)')
fig.show()

#print(world.columns[2:])
#print(world.iloc[:,2:].values[0])

fig=px.line(x=world.columns[2:], y=world.iloc[:,2:].values[0],
            title="World Forest Area in sq. m.", template='plotly_dark')
fig.update_layout(xaxis_title='Years',
                  yaxis_title='Forest Land Area in sq. m.')
fig.show()

#print(regions)

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

fig.update_layout(title='Regions Forest Area in sq. m.',
                   xaxis_title='Years',
                   yaxis_title='Forest Land Area in sq. m.',
                  template='plotly_dark')
fig.show()

regije_bar=[]
regije_bar_2018=[]
regije_bar_1990=[]

for i in code_name.keys():
    regija=regions[regions['Country Code']==i]
    regije_bar.append(code_name[i])
    regije_bar_2018.append(regija['2018'].values[0])
    regije_bar_1990.append(regija['1990'].values[0])

#print(regije_bar)
#print(regije_bar_2018)
#print(regije_bar_1990)

fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Bar(x=regije_bar, y=regije_bar_1990,name='1990'),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=regije_bar, y=regije_bar_2018,name='2018'),
    row=1, col=2
)

fig.update_layout(title_text="Regions Forest Area in sq. m.",
                  template='plotly_dark')
fig.show()
