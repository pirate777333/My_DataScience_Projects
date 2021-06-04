import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from math import sqrt
from scipy.stats import boxcox
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

olimpijske='D:/udemy_kurs_py/DataScienceProjects/Project95/Pandas_Project_Session/data/olympics/olympics.csv'
drzave='D:/udemy_kurs_py/DataScienceProjects/Project95/Pandas_Project_Session/data/olympics/Summer Olympic medallists 1896 to 2008 - IOC COUNTRY CODES.csv'

dfoli=pd.read_csv(olimpijske,skiprows=[0,1,2,3])
dfdrzave=pd.read_csv(drzave)

#print(dfoli.head())
#print(dfdrzave.head())

print(dfoli.head())
print(dfoli.columns)
print(dfoli.dtypes)
print(dfoli.tail())
print(dfoli.shape)
print(dfoli.describe())
print(dfoli.isnull().sum())
print(dfoli['Event_gender'].unique())
print(dfoli['Sport'].unique())

ef = dfoli[dfoli['Athlete']=='FLACK, Edwin']
print(ef)

athl_m=dfoli[(dfoli['Sport']=='Athletics')&(dfoli['Event_gender']=='M')&(dfoli['Medal']=='Gold')]
print(athl_m.groupby('NOC')['Medal'].count())
print(athl_m.groupby(['NOC','Athlete'])['Medal'].count())

ind=dfoli[(dfoli['NOC']=='IND')&(dfoli['Event_gender']=='M')&(dfoli['Medal']=='Gold')]
print(ind)

indw=dfoli[(dfoli['NOC']=='IND')&(dfoli['Event_gender']=='W')]
print(indw)

recent=dfoli[dfoli['Edition']>=2000]
big5=dfoli.groupby('NOC')['Medal'].count().reset_index()
big5=big5.sort_values(by='Medal',ascending=False)

fig=px.bar(big5.iloc[:5,:],x='NOC',y='Medal',title='Big 5 Recent Years',template='plotly_dark')
fig.show()

gmh=dfoli[(dfoli['Sport']=='Hockey')&(dfoli['Event_gender']=='M')&(dfoli['Medal']=='Gold')]
gmh=gmh.sort_values(by='Edition',ascending=False)
print(gmh)

wg=dfoli[(dfoli['Medal']=='Gold')&(dfoli['Event_gender']=='W')]
print(wg.shape[0])

yearly=dfoli.groupby(['Edition','NOC'])['Medal'].count().reset_index()
yearly.fillna(0,inplace=True)
#print(yearly.head(49))

pivot = pd.pivot_table(yearly, values="Medal", index=["Edition"], columns=["NOC"], aggfunc=np.sum)
pivot = pivot.cumsum()
pivot.fillna(method='ffill',inplace=True)
pivot.fillna(0,inplace=True)
print(pivot)

city=dfoli.drop_duplicates(subset=['City', 'Edition'], keep='first')
city=city.groupby('City')['Edition'].count().reset_index()
city=city.sort_values(by='Edition',ascending=False)
#print(city)
fig=px.bar(city.iloc[:5,:],x='City',y='Edition',color='City',
                     title='City',template='plotly_dark')
fig.show()

athlete=dfoli.groupby('Athlete')['Medal'].count().reset_index()
athlete=athlete.sort_values(by='Medal',ascending=False)
fig=px.bar(athlete.iloc[:5,:],x='Athlete',y='Medal',color='Athlete',
                     title='Athlete',template='plotly_dark')
fig.show()

zlato=dfoli[dfoli['Medal']=='Gold']
athlete=zlato.groupby('Athlete')['Medal'].count().reset_index()
athlete=athlete.sort_values(by='Medal',ascending=False)
fig=px.bar(athlete.iloc[:5,:],x='Athlete',y='Medal',color='Athlete',
                     title='Athlete',template='plotly_dark')
fig.show()

flattened = pd.DataFrame(pivot.to_records())
drzave=flattened.columns[1:]
print(drzave)
godine=flattened.Edition.values
print(godine)
new=flattened.T
new=new.iloc[1:,:]
new.columns=godine
new['drzave']=drzave
print(new)

df_long = pd.melt(new, id_vars=['drzave'], value_vars=godine)
fig = px.choropleth(df_long,                            # Input Dataframe
                     locations="drzave",           # identify country code column
                     color="value",                     # identify representing column
                     hover_name="drzave",              # identify hover name
                     animation_frame="variable",        # identify date column
                     projection="natural earth",        # select projection
                     color_continuous_scale = 'Peach',  # select prefer color scale
                     range_color=[0,4500]              # select range of dataset
                     )        
fig.show()          
#fig.write_html("example_map.html")                      # write the result to HTML file
