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

dftrain=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project92/train.csv')
dftest=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project92/test.csv')

print(dftrain.head())
print(dftrain.columns)
print(dftrain.dtypes)
print(dftrain.tail())
print(dftrain.shape)
print(dftrain.describe())
print(dftrain.isnull().sum())
#print(dftrain.Credit_History.unique())

sns.countplot(dftrain['Department'])
plt.show()

sns.countplot(dftrain['salary'])
plt.show()

sns.distplot(dftrain['satisfaction_level'])
plt.show()

a=pd.get_dummies(dftrain['salary'])
lc=[]
for i in a.columns:
    lc.append('salary_'+i)
a.columns=lc
print(a)
b=pd.get_dummies(dftrain['Department'])
lc=[]
for i in b.columns:
    lc.append('Department_'+i)
b.columns=lc
print(b)

dftrain.drop(['salary','Department'],axis=1,inplace=True)
dftrain=pd.concat([dftrain,a],axis=1)
dftrain=pd.concat([dftrain,b],axis=1)
print(dftrain.columns)

X=dftrain.drop(['Attrition'],axis=1)
y=dftrain['Attrition']

print(X)
print(y)

from sklearn.ensemble import RandomForestClassifier
regr = RandomForestClassifier()
regr.fit(X, y)

a=pd.get_dummies(dftest['salary'])
lc=[]
for i in a.columns:
    lc.append('salary_'+i)
a.columns=lc
print(a)
b=pd.get_dummies(dftest['Department'])
lc=[]
for i in b.columns:
    lc.append('Department_'+i)
b.columns=lc
print(b)

dftest.drop(['salary','Department'],axis=1,inplace=True)
dftest=pd.concat([dftest,a],axis=1)
dftest=pd.concat([dftest,b],axis=1)
print(dftest.columns)

Xt=dftest.drop(['Attrition','ID'],axis=1)
yt=dftest['Attrition']
print(Xt.columns)
print(X.columns)
predicted = regr.predict(Xt)
matrix = confusion_matrix(yt, predicted)

print(matrix)
print(accuracy_score(yt, predicted))
