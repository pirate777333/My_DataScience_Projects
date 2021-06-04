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

def feedback(text):
    if text>8.0:
        return 'Good'
    else:
        return 'Average'

def runtime(text):
    if text<90:
        return "short"
    elif text<120:
        return "average"
    else:
        return "long"

df=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project95/Pandas_Project_Session/data/movies/IMDB_Movie_Data.csv')
df.fillna(df.mean(),inplace=True)
print(df.head())
print(df.columns)
print(df.dtypes)
print(df.tail())
print(df.shape)
print(df.describe())
print(df.isnull().sum())
#print(dfoli['Event_gender'].unique())
#print(df['Director'].unique())

corr=df.corr()
sns.heatmap(corr,annot=True)
plt.show()

multi=df[((df['Director']=='James Gunn')|(df['Director']=='Ridley Scott'))&(df['Rating']>=8.0)]
print(multi)
print(df[df['Director']=='Ridley Scott'])
print(df[df['Rating']>=8.0])
print(df[df['Rating']>=8.0].sort_values(by='Rating',ascending=False))

multi2=df[((df['Year']>2006)&(df['Year']<2010))&(df['Rating']>=8.0)&(df['Revenue (Millions)']<df['Revenue (Millions)'].mean())]
print(multi2)
multi3=df[((df['Year']>2006)&(df['Year']<2010))&(df['Rating']>=8.0)&(df['Revenue (Millions)']<df['Revenue (Millions)'].quantile(q=0.25))]
print(multi3)
df['Feedback']=df['Rating'].apply(feedback)
print(df.head())

genre=df.groupby('Genre')['Rating'].mean().reset_index()
genre=genre.sort_values(by='Rating',ascending=False)
fig=px.bar(genre,x='Genre',y='Rating',color='Genre',
           title='Best Genres',template='plotly_dark')
fig.show()

best=df.sort_values(by='Rating',ascending=False)
fig=px.bar(best.iloc[:5,:],x='Title',y='Rating',color='Title',
           title='Best 5',template='plotly_dark')
fig.show()

worst=df.sort_values(by='Rating',ascending=True)
fig=px.bar(worst.iloc[:5,:],x='Title',y='Rating',color='Title',
           title='Worst 5',template='plotly_dark')
fig.show()

year=df.groupby('Year')['Rating'].mean().reset_index()
year=year.sort_values(by='Rating',ascending=False)
fig=px.bar(year,x='Year',y='Rating',color='Year',
           title='Best Years',template='plotly_dark')
fig.show()

df['runtime']=df['Runtime (Minutes)'].apply(runtime)
run=df.groupby('runtime')['Rating'].mean().reset_index()
run=run.sort_values(by='Rating', ascending=False)
fig=px.bar(run,x='runtime',y='Rating',color='runtime',
           title='Runtime',template='plotly_dark')
fig.show()
