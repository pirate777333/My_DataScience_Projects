import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import HeatMap
import folium

def education(text):
    if text==1:
        return 'Undergrad'
    if text==2:
        return 'Graduate'
    if text==3:
        return 'Advanced'

def accounts(r):
    if (r['Securities Account']==1)&(r['CD Account']==1):
        return 'Both'
    elif (r['Securities Account']==1)&(r['CD Account']==0):
        return 'Secc'
    elif (r['Securities Account']==0)&(r['CD Account']==1):
        return 'Cd'
    elif (r['Securities Account']==0)&(r['CD Account']==0):
        return 'Nothing'    
# LOAD FILE

df=pd.read_excel('D:/udemy_kurs_py/DataScienceProjects/Project3/Bank_Personal_Loan_Modelling.xlsx',sheet_name=1)
####df['date']=pd.to_datetime(df['date'],format='%m/%d/%Y')
####df['Month']=df['date'].dt.month
####df['Day']=df['date'].dt.day
####df['Weekday']=df['date'].dt.day_name()
##df.fillna(0,inplace=True)
##df=df[(df['babies']!=0)&(df['adults']!=0)&(df['babies']!=0)]
##all_data=df.copy()
##all_cancel=df[df['is_canceled']==1]
##df=df[df['is_canceled']==0]
df=df.drop('ID',axis=1)
df=df.drop('ZIP Code',axis=1)
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
print(df.describe())
print(df[['Personal Loan','Education','Income']].describe())
print(df.info())

# VISUALIZE

fig=px.box(df,y=['Age','Experience','Income','Family','Education'])
fig.show()

df.hist()
plt.show()

# EXPERIENCE COLUMN ANALYSIS

df=df[df['Experience']>0]

fig=px.box(df,y='Experience',
           template='plotly_dark')
fig.show()

print(df['Experience'].unique())

plt.hist(df['Experience'])
plt.show()

# CORRELATION
corr=df.corr(method ='pearson')
print(df.corr(method ='pearson'))

sns.heatmap(corr, annot=True)
plt.show()

df=df.drop('Experience',axis=1)

# EDUCATION

plt.hist(df['Education'])
plt.show()
print(df['Education'].unique())

df['Education_Mark']=df['Education'].apply(education)

fig=px.pie(df, values='Education', names='Education_Mark',
       title='Education')
fig.show()

# ACCOUNT HOLDERS

secacc=df.groupby('Securities Account')['Age'].count().reset_index()
fig=px.bar(secacc,x='Securities Account',y='Age',color='Securities Account',
           template='plotly_dark',title='Securities Account')
fig.show()

cdacc=df.groupby('CD Account')['Age'].count().reset_index()
fig=px.bar(cdacc,x='CD Account',y='Age',color='CD Account',
           template='plotly_dark',title='Securities Account')
fig.show()

df['Acc_Holder']=df.apply(accounts, axis=1)
accs=df.groupby('Acc_Holder')['Age'].count().reset_index()
fig=px.bar(accs,x='Acc_Holder',y='Age',color='Acc_Holder',
           template='plotly_dark',title='Securities Account')
fig.show()
fig=px.pie(df, values='Age', names='Acc_Holder',
       title='Acc_Holder')
fig.show()

# EDUCATION/INCOME/PERSONAL LOAN

edu_inc=df.groupby(['Education','Personal Loan'])['Income'].count().reset_index()
fig=px.bar(edu_inc,x='Education',y='Income',color='Personal Loan',
           template='plotly_dark',title='Education and Income')
fig.show()

fig=px.box(df,x='Education',y='Income',facet_col='Personal Loan',
           template='plotly_dark',title='Education and Income and Personal Loans')
fig.show()

# EXTRA ANALYSIS
lista=['CreditCard', 'Online', 'Securities Account']

for i in lista:
    df2=df.groupby(['Acc_Holder',i])['Age'].count().reset_index()
    fig=px.bar(df2,x='Acc_Holder',y='Age',color=i,
               template='plotly_dark',title=i)
    fig.show()

# AGE VS LOAN

fig=px.box(df,x='Personal Loan',y='Age')
fig.show()

# FAMILY VS LOAN

fig=px.box(df,x='Personal Loan',y='Family')
fig.show()

# INCOME VS LOAN

fig=px.box(df,x='Personal Loan',y='Income')
fig.show()
