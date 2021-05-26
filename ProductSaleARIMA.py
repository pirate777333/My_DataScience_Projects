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
from math import sqrt
from scipy.stats import boxcox
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

def splitStringGODINA(text):
    g=text.split('-').strip()[0]
    return g

def splitStringMJESEC(text):
    m=text.split('-').strip()[1]
    return m    

df=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project9/perrin-freres-monthly-champagne-.csv')
#df['Month']=pd.to_datetime(df['Month'], format='%Y-%m')
df.dropna(inplace=True)
#df['Year']=df['Month'].apply(splitStringGODINA)
#df['Month_D']=df['Month'].apply(splitStringMJESEC)
df['Month']=pd.to_datetime(df['Month'], format='%Y-%m')
df.columns=['Date','Sale']
df = df.set_index('Date')
##print(df.head())
##print(df.columns)
##print(df.shape)
##print(df.dtypes)
##print(df.describe())
##print(df.isnull().sum())
##print(df['Month'].unique())

fig = px.box(df, y="Sale", template='plotly_dark')
fig.show()

fig=px.line(y=df['Sale'],x=df.index,
       template='plotly_dark',title='Sales over months',
       labels=dict(x="Date", y="Sale"))
fig.show()

result=adfuller(df['Sale'],autolag='AIC')
print(result[1])
#result2=kpss(df['Sale'])
#print(result2[1])

df['Shift_1']=df['Sale']-df['Sale'].shift(1)
df['Shift_12']=df['Sale']-df['Sale'].shift(12)

result=adfuller(df['Shift_12'].dropna(),autolag='AIC')
print(result[1])
#result2=kpss(df['Shift_12'].dropna())
#print(result2[1])

fig=px.line(y=df['Shift_12'],x=df.index,
       template='plotly_dark',title='Sales over months',
       labels=dict(x="Date", y="Sale"))
fig.show()

# p,d,q (pacf,diff,acf)

# q
plot_acf(df['Shift_12'].iloc[13:], ax=plt.gca(), lags=30)
plt.show()

# p
plot_pacf(df['Shift_12'].iloc[13:], ax=plt.gca(), lags=30)
plt.show()

order=(1,1,1)

model=ARIMA(df['Sale'],order=order)
model=model.fit()
print(model.summary())

df['forecast']=model.predict(start=90,end=104,dynamic=True)

df[['Sale','forecast']].plot()
plt.show()

model2 = SARIMAX(df['Sale'], order=order, seasonal_order=(1,1,1,12))

modelfit=model2.fit()
print(modelfit.summary())

df['forecast_s']=modelfit.predict(start=90,end=104,dynamic=True)

df[['Sale','forecast_s']].plot()
plt.show()

future_dates=[df.index[-1]+DateOffset(months=x) for x in range(0,24)]
dataset=pd.DataFrame(index=future_dates[1:],columns=df.columns)

futureDF=pd.concat([df,dataset])

futureDF['forecast_s2']=modelfit.predict(start=104,end=124,dynamic=True)

futureDF[['Sale','forecast_s2']].plot()
plt.show()

#pred=modelfit.forecast(7)
#pred_df=pred.to_frame()
#print(pred)


##df=pd.read_csv('D:/rektorov_rad/programirano/Prediction_Arima/00b03bf1-e424-471c-bafb-196f8204e4d9_Data.csv')
##
###print(df.iloc[:-5,2:-1])
##df=df.iloc[:-5,2:-1]
##
##stupci=['Country Name','Country Code']
##for i in df.columns[2:]:
##    stupac=i.split(' ')[0]
##    stupci.append(stupac)
##
###print(stupci)
##df.columns=stupci
##
##def stringops(text):
##    if text=='..':
##        return None
##    else:
##        return text
##
##for i in df.columns:
##    df[i]=df[i].apply(stringops)
##
##df=df.dropna()
##
##for i in df.columns[2:]:
##    df[i]=pd.to_numeric(df[i])
##
###print(df.isnull().sum())
###print(df.dtypes)
###print(df.shape)
##
##croatia=df[df['Country Name']=='World'] # World or Croatia
##croatia=croatia.T
##croatia=croatia.iloc[2:,:]
###print(croatia)
###print(croatia.shape)
##year=croatia.index.tolist()
##data=croatia.iloc[:,0].tolist()
####fig=px.line(y=data,x=year,
####       template='plotly_dark',title='Urban population',
####       labels=dict(x="Years", y="Urban population"))
####fig.show()
##
##croatia.index = pd.to_datetime(croatia.index)
##croatia.iloc[:,0]=pd.to_numeric(croatia.iloc[:,0],downcast='float')
##croatia.columns=['urban_pop']
##world=croatia.copy()
##
####decomposition=sm.tsa.seasonal_decompose(croatia.iloc[:,0],model='additive')
####decomposition.plot()
####plt.show()
##
####length_train=48
####train=croatia.iloc[:length_train,:]
####test=croatia.iloc[length_train:,:]
#####print(train.shape)
#####print(test.shape)
#####print(train)
####train.index = pd.to_datetime(train.index)
####train.iloc[:,0]=pd.to_numeric(train.iloc[:,0],downcast='float')
####train.columns=['urban_pop']
#####print(test)
####test.index = pd.to_datetime(test.index)
####test.iloc[:,0]=pd.to_numeric(test.iloc[:,0],downcast='float')
####test.columns=['urban_pop']
###print(train.dtypes)
###print(test.dtypes)
### MANJE OD 0.05 JE STATIONARY
##result=adfuller(croatia.iloc[:,0],autolag='AIC')
##print(result[1])
##
##croatia=pd.DataFrame(boxcox(croatia.iloc[:,0],lmbda=0),index=croatia.index)
##result=adfuller(croatia.iloc[:,0],autolag='AIC')
##print(result[1])
##
##year=croatia.index.tolist()
##data=croatia.iloc[:,0].tolist()
####fig=px.line(y=data,x=year,
####       template='plotly_dark',title='Urban population',
####       labels=dict(x="Years", y="Urban population"))
####fig.show()
##
####croatia=pd.DataFrame(croatia-croatia.shift(periods=1),index=croatia.index)
####croatia.dropna(inplace=True)
####print(croatia)
####result=adfuller(croatia.iloc[:,0],autolag='AIC')
####print(result[1])
####year=croatia.index.tolist()
####data=croatia.iloc[:,0].tolist()
####fig=px.line(y=data,x=year,
####       template='plotly_dark',title='Urban population',
####       labels=dict(x="Years", y="Urban population"))
####fig.show()
##
##result2=kpss(croatia.iloc[:,0])
##print(result2[1])
##
##croatia.iloc[:,0]=pd.to_numeric(croatia.iloc[:,0],downcast='float')
##croatia.columns=['urban_pop']
##
####plot_acf(croatia, ax=plt.gca(), lags=10)
####plt.show()
####
####plot_pacf(croatia, ax=plt.gca(), lags=20)
####plt.show()
##
##length_train=48
##train=croatia.iloc[:length_train,:]
##test=croatia.iloc[length_train:,:]
###print(train.shape)
###print(test.shape)
###print(train)
##train.index = pd.to_datetime(train.index)
##train.iloc[:,0]=pd.to_numeric(train.iloc[:,0],downcast='float')
##train.columns=['urban_pop']
###print(test)
##test.index = pd.to_datetime(test.index)
##test.iloc[:,0]=pd.to_numeric(test.iloc[:,0],downcast='float')
##test.columns=['urban_pop']
##
##stepwise_fit=auto_arima(croatia.iloc[:,0],trace=True,suppress_warnings=True,
##                        start_p=0, d=None, start_q=0, max_p=5, max_d=5, max_q=5)
###print(stepwise_fit.summary())
##
##model=ARIMA(train.iloc[:,:],order=(0,2,0))
##model=model.fit()
###print(model.summary())
##
##start=len(train)
##end=len(train)+len(test)-1
##pred=model.predict(start=start,end=end,typ='levels')
###print(pred)
##
###pred.index=df.index[start:end+1]
##plt.plot(train,label='train')
##plt.plot(test,label='test')
##plt.plot(pred,label='pred')
##plt.legend()
##plt.show()
##
##rmse=sqrt(mean_squared_error(pred,test.iloc[:,0]))
##print(test.iloc[:,0].mean())
##print(rmse)
##
##
##model2=ARIMA(croatia.iloc[:,:],order=(0,2,0))
##model2=model2.fit()
###print(model.summary())
###print(croatia.tail())
##
##index_future_dates=pd.date_range(start='2019-01-01',end='2031-01-01',freq='YS')
###print(index_future_dates)
##pred=model2.predict(start=len(croatia),end=len(croatia)+12,typ='levels').rename('ARIMA Preds')
##pred.index=index_future_dates
##print(pred)
##plt.plot(croatia,label='History')
##plt.plot(pred,label='Future')
##plt.legend()
##plt.show()
##
###Function
##def invboxcox(y,ld):
##   if ld == 0:
##      return(np.exp(y))
##   else:
##      return(np.exp(np.log(ld*y+1)/ld))
##
### Test the code
##predictions_df=pred.to_frame()
##predictions_df=predictions_df.iloc[:,0]
###print(x)
##print(np.exp(predictions_df))
##predictions_df=(np.exp(predictions_df)).to_frame()
####ld = 0
####y = stats.boxcox(x,ld)
####print (invboxcox(y[0],ld))
##
##plt.plot(world,label='History')
##plt.plot(predictions_df,label='Future')
##plt.legend()
##plt.show()
##
##
##fig = go.Figure()
##
##fig.add_trace(go.Scatter(
##    x=world.index,
##    y=world.iloc[:,0],
##    name = 'History',
##    mode='lines'
##))
##fig.add_trace(go.Scatter(
##    x=predictions_df.index,
##    y=predictions_df.iloc[:,0],
##    name='Prediction',
##    mode='lines'       
##))
##fig.update_layout(template='plotly_dark', title='World Urban Population')
##fig.show()
