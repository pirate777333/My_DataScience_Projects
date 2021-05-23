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

# LOAD DATA

deliveries=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project6/deliveries.csv')

matches=pd.read_csv('D:/udemy_kurs_py/DataScienceProjects/Project6/matches.csv')
##print(deliveries.head())
##print(deliveries.tail())
##print(deliveries.isnull().sum())
##print(deliveries.dtypes)
##print(deliveries.describe())
##print(deliveries.shape)
##print(deliveries.columns)
##
### DA Warner Analysis
##
##DA_Warner=deliveries[deliveries['batsman']=='DA Warner']
##print(DA_Warner)
##
##dismissal=DA_Warner.groupby('dismissal_kind')['match_id'].count().reset_index()
##
##fig = px.pie(dismissal, values='match_id', names='dismissal_kind',
##             title='Dismissal kind for DA Warner',template='plotly_dark')
##fig.show()
##
##runs=DA_Warner.groupby('batsman_runs')['match_id'].count().reset_index()
##
##fig = px.pie(runs, values='match_id', names='batsman_runs',
##             title='Runs for DA Warner',template='plotly_dark')
##fig.show()
##
### TOTAL RUNS
##
##total=deliveries.sort_values(by='total_runs', ascending=False)
##
##fig=px.bar(total.iloc[:10,:], x='batting_team', y='total_runs',
##           title='Total runs',template='plotly_dark')
##fig.show()
##
### BATSMAN
##
##b1=deliveries.groupby('batsman')['ball'].sum().reset_index()
##b1=b1.sort_values(by='ball', ascending=False)
##fig=px.bar(b1.iloc[:10,:], x='batsman', y='ball', color='batsman',
##           title='Ball',template='plotly_dark')
##fig.show()
##b2=deliveries.groupby('batsman')['batsman_runs'].sum().reset_index()
##b2=b2.sort_values(by='batsman_runs', ascending=False)
##fig=px.bar(b2.iloc[:10,:], x='batsman', y='batsman_runs', color='batsman',
##           title='Batsman runs',template='plotly_dark')
##fig.show()
##
### DISMISSALS
##
##dismiss=deliveries.groupby('dismissal_kind')['match_id'].count().reset_index()
##
##fig = px.pie(dismiss, values='match_id', names='dismissal_kind',
##             title='Dismissal kinds',template='plotly_dark')
##fig.show()
##
##print(matches.head())
##print(matches.tail())
##print(matches.isnull().sum())
##print(matches.dtypes)
##print(matches.describe())
##print(matches.shape)
##print(matches.columns)
##
### ANALYSIS BY MATCHES
##
##matches=matches.drop('umpire3', axis=1)
##
##matches_season=matches.groupby('season')['id'].count().reset_index()
##fig=px.bar(matches_season, x='season', y='id', color='season',
##           title='Total Matches',template='plotly_dark')
##fig.show()
##
##cities=matches.groupby('city')['id'].count().reset_index()
##fig=px.pie(cities, values='id', names='city',
##           title='Total Cities',template='plotly_dark')
##fig.show()
##
##total_teams1=matches.team1.unique()
##total_teams2=matches.team2.unique()
##
##for i in total_teams2:
##    if i not in total_teams1:
##        total_teams1.append(i)
##
##print(len(total_teams1))
##
##motm = matches.groupby('player_of_match')['id'].count().reset_index()
##motm=motm.sort_values(by='id',ascending=False)
##fig=px.bar(motm.iloc[:5,:], x='player_of_match', y='id', color='player_of_match',
##           title='MOTM',template='plotly_dark')
##fig.show()
##
##wbr=matches.sort_values(by='win_by_runs',ascending=False)
##fig=px.bar(wbr.iloc[:3,:], x='winner', y='win_by_runs', color='winner',
##           title='Win by runs',template='plotly_dark')
##fig.show()
##
##wbw=matches.sort_values(by='win_by_wickets',ascending=False)
##fig=px.bar(wbw.iloc[:5,:], x='winner', y='win_by_wickets', color='winner',
##           title='Win by wickets',template='plotly_dark')
##fig.show()
##
##seasons=matches.groupby(['season','toss_decision'])['id'].count().reset_index()
##
##fig=px.bar(seasons, x='season', y='id', color='toss_decision',
##           title='Toss Decision',template='plotly_dark')
##fig.show()
##
##tosswinners=matches.groupby('toss_winner')['id'].count().reset_index()
##tosswinners=tosswinners.sort_values(by='id',ascending=False)
##fig=px.bar(tosswinners.iloc[:5,:], x='toss_winner', y='id', color='toss_winner',
##           title='Toss Winners',template='plotly_dark')
##fig.show()
##
##TeamMatchWin=pd.DataFrame(columns=['Team', 'Matches', 'Wins'])
##
##for team in total_teams1:
##    dfTeam=matches[(matches['team1']==team)|(matches['team2']==team)]
##    totalMatches=dfTeam.shape[0]
##    dfWin=matches[matches['winner']==team]
##    totalWins=dfWin.shape[0]
##    TeamMatchWin=TeamMatchWin.append({'Team':team, 'Matches':totalMatches,
##                                      'Wins':totalWins}, ignore_index=True)
##print(TeamMatchWin)
##fig = go.Figure(data=[
##    go.Bar(name='Total Matches', x=TeamMatchWin.Team, y=TeamMatchWin.Matches),
##    go.Bar(name='Total Wins', x=TeamMatchWin.Team, y=TeamMatchWin.Wins)
##])
### Change the bar mode
##fig.update_layout(barmode='group')
##fig.show()
##
### MERGE AND ANALYSE
##
##print(deliveries.head())
##print(matches.head())
##print(deliveries.columns)
##print(matches.columns)

season=deliveries.merge(matches,left_on='match_id', right_on='id',how='left').drop('id', axis=1)
print(season.head())
print(season.columns)

seasonalruns=season.groupby('season')['total_runs'].sum().reset_index()
fig=px.bar(seasonalruns, x='season', y='total_runs', color='season',
           title='Runs per season',template='plotly_dark')
fig.show()

runsmatches=season.groupby(['season','match_id'])['total_runs'].sum().reset_index()
seasonsn=runsmatches.season.unique()

smr=pd.DataFrame(columns=['Season','Matches','Runs'])

for seas in seasonsn:
    data=runsmatches[runsmatches['season']==seas]
    Nmatch=data.shape[0]
    TotRun=data['total_runs'].sum()
    smr=smr.append({'Season':seas, 'Matches':Nmatch,
                    'Runs':TotRun}, ignore_index=True)

print(smr)
smr['AvgRunPerMatch']=smr['Runs']/smr['Matches']

fig=px.line(x=smr['Season'],y=smr['AvgRunPerMatch'],
            title='Average Runs per season',template='plotly_dark')
fig.show()

lucky=season[season['team1']==season['winner']]
lucky1=lucky.groupby('city')['match_id'].count().reset_index()
fig=px.pie(lucky1, values='match_id', names='city',
           title='Lucky Cities',template='plotly_dark')
fig.show()

lucky2=lucky.groupby('venue')['match_id'].count().reset_index()
fig=px.pie(lucky2.iloc[:5,:], values='match_id', names='venue',
           title='Lucky Venues',template='plotly_dark')
fig.show()

tworandoms=season['winner'].unique()
t1=tworandoms[2]
t2=tworandoms[0]

twodf=season[(season['winner']==t1)|(season['winner']==t2)]

twodf2=twodf.groupby('winner')['match_id'].count().reset_index()
fig=px.pie(twodf2, values='match_id', names='winner',
           title='Compare 2 teams',template='plotly_dark')
fig.show()

twodf3=twodf.groupby(['season','winner'])['match_id'].count().reset_index()
fig=px.bar(twodf3, x='season', y='match_id', color='winner',
           title='Compare 2 teams over seasons',template='plotly_dark')
fig.show()
