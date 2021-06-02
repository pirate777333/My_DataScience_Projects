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


df=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project91/Loan Prediction Dataset.csv')
df['LoanAmount'].fillna((df['LoanAmount'].mean()), inplace=True)
df['Loan_Amount_Term'].fillna((df['Loan_Amount_Term'].mean()), inplace=True)

df = df[df['Credit_History'].notna()]

df['Gender'].fillna((df['Gender'].mode()[0]), inplace=True)
df['Married'].fillna((df['Married'].mode()[0]), inplace=True)
df['Dependents'].fillna((df['Dependents'].mode()[0]), inplace=True)
df['Self_Employed'].fillna((df['Self_Employed'].mode()[0]), inplace=True)

df['Total']=df['ApplicantIncome']+df['CoapplicantIncome']

print(df.head())
print(df.columns)
print(df.dtypes)
print(df.tail())
print(df.shape)
print(df.describe())
print(df.isnull().sum())
print(df.Credit_History.unique())

sns.countplot(df['Gender'])
plt.show()

df['Married'].hist()
plt.show()

df['LoanAmount'].hist()
plt.show()

sns.distplot(df['Loan_Amount_Term'])
plt.show()

corr=df.corr()
sns.heatmap(corr,annot=True)
plt.show()

##alla=['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
##       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
##       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status',
##       'Total']

need=['Gender','Married','Dependents','Education','Property_Area',
      'Loan_Status','Self_Employed']

dfn=df[need]

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

for n in need:
    dfn[n]=le.fit_transform(dfn[n])

print(dfn)

X=dfn.drop(columns=['Loan_Status'],axis=1)
y=dfn['Loan_Status']

from sklearn.model_selection import train_test_split

xtr,xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()

model.fit(xtr,ytr)

print(model.score(X,y))

ypred=model.predict(xte)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(yte,ypred)

print(cm)

sns.heatmap(cm,annot=True)
plt.show()

##from sklearn.model_selection import GridSearchCV
### defining parameter range 
##param_grid = {'C': [0.1, 1, 10, 100],  
##              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
##              'gamma':['scale', 'auto'],
##              'kernel': ['linear']}  
##   
##grid = GridSearchCV(model, param_grid, refit = True, verbose = 3,n_jobs=-1) 
##   
### fitting the model for grid search 
##grid.fit(X_train, y_train) 
## 
### print best parameter after tuning 
##print(grid.best_params_) 
##grid_predictions = grid.predict(X_test) 
##   
### print classification report 
##print(classification_report(y_test, grid_predictions)) 

##from sklearn.model_selection import GridSearchCV
##from sklearn.neighbors import KNeighborsClassifier
##kn = KNeighborsClassifier()
##params = {
##    'n_neighbors' : [5, 25],
##    'weights': ['uniform', 'distance'],
##    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
##}
##grid_kn = GridSearchCV(estimator = kn,
##                        param_grid = params,
##                        scoring = 'accuracy', 
##                        cv = 5, 
##                        verbose = 1,
##                        n_jobs = -1)
##grid_kn.fit(X_train, y_train)
