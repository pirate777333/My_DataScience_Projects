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

df_fpla=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/ProjectForests/194d83ff-0135-4554-b1d0-2ecadf660b00_Data.csv')
df_fpla=df_fpla.iloc[:-5,2:-2]
####print(df_fpla.head())
####print(df_fpla.tail(15))
####print(df_fpla.shape)
####print(df_fpla.columns)
####print(df_fpla.isnull().sum())
####print(df_fpla.dtypes)

stupci=['Country Name','Country Code']
for i in range(1960,2019):
    stupci.append(str(i))

df_fpla.columns=stupci

for i in df_fpla.columns:
    df_fpla[i]=df_fpla[i].apply(stringops)
    
df_fpla=df_fpla[['Country Name', 'Country Code','1990', '1991',
       '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000',
       '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009',
       '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']]

countries=df_fpla.iloc[:-8,:]
world=df_fpla[df_fpla['Country Name']=='World']
world=world.iloc[:,:-2]
regions=df_fpla.iloc[-8:,:]

countries=countries.dropna()
for i in countries.columns[2:]:
    countries[i]=pd.to_numeric(countries[i])
for i in world.columns[2:]:
    world[i]=pd.to_numeric(world[i])
for i in regions.columns[2:]:
    regions[i]=pd.to_numeric(regions[i])

regions=regions[regions['Country Name']!='World']

croatia=countries[countries['Country Name']=='Croatia']

fig=px.line(x=croatia.columns[2:], y=croatia.iloc[:,2:].values[0],
            title="Croatia Forest Area in %", template='plotly_dark')
fig.update_layout(xaxis_title='Years',
                  yaxis_title='Forest Land Area in %')
fig.show()

#print(countries)
#print(world)
#print(regions)

fig=px.line(x=world.columns[2:], y=world.iloc[:,2:].values[0],
            title="World Forest Area in %", template='plotly_dark')
fig.update_layout(xaxis_title='Years',
                  yaxis_title='Forest Land Area in %')
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

bestch=countries.sort_values(by='Difference',ascending=False)
fig=px.bar(x=bestch.iloc[:10,0],y=bestch.iloc[:10,-1],color=bestch.iloc[:10,0],
           title='Best 10 in positive change',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Forest Land Area in sq. m. (change 2018-1990)')
fig.show()

bestch=countries.sort_values(by='Difference',ascending=True)
fig=px.bar(x=bestch.iloc[:10,0],y=bestch.iloc[:10,-1],color=bestch.iloc[:10,0],
           title='Worst 10 in change',template='plotly_dark')
fig.update_layout(xaxis_title='Countries',
                  yaxis_title='Forest Land Area in sq. m. (change 2018-1990)')
fig.show()

countries_px=countries[['Country Name','Country Code','2018']]
print(countries_px)

fig = go.Figure(data=go.Choropleth(
    locations=countries_px['Country Code'],
    z=countries_px['2018'].astype(float),
    colorscale='Greens',
    autocolorscale=False,
    text=countries_px['Country Name'], # hover text
    marker_line_color='green', # line markers between states
    colorbar_title="Forest Land Area in %"
))

fig.update_layout(
    title_text='Forest Land Area in %',
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

world=world.T
world=world.iloc[2:,:]

year=world.index.tolist()
data=world.iloc[:,0].tolist()

world.index = pd.to_datetime(world.index)
world.iloc[:,0]=pd.to_numeric(world.iloc[:,0],downcast='float')
world.columns=['forest']

#print(world)

result=adfuller(world.iloc[:,0],autolag='AIC')
print(result[1])

plot_acf(world, ax=plt.gca(), lags=10)
plt.show()

plot_pacf(world, ax=plt.gca(), lags=10)
plt.show()

#stepwise_fit=auto_arima(world.iloc[:,0],trace=True,suppress_warnings=True,
#                        start_p=0, d=None, start_q=0, max_p=4, max_d=2, max_q=4)
#print(stepwise_fit.summary())
#2,1,1
#2,0,1
#0,1,0
model=ARIMA(world.iloc[:,:],order=(0,1,0))
model=model.fit()

index_future_dates=pd.date_range(start='2016-01-01',end='2026-01-01',freq='YS')
pred=model.predict(start=len(world),end=len(world)+10,typ='levels').rename('ARIMA Preds')
pred.index=index_future_dates
#print(pred)
indeksi=pred.index
predvidanja=pred.to_frame()
lista_pred=predvidanja.iloc[:,0].values
#print(indeksi)
#print(lista_pred)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=world.index,
    y=world.iloc[:,0],
    name = 'History',
    mode='lines'
))
fig.add_trace(go.Scatter(
    x=indeksi,
    y=lista_pred,
    name='Prediction',
    mode='lines'       
))
fig.update_layout(template='plotly_dark', title='Forest Land Area in %')
fig.show()
