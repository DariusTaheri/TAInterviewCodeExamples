# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:56:59 2022

@author: DariusPC
"""

from MyFunctions import dfinfo, ScriptVars
import pandas as pd
import numpy as np
from matplotlib import pyplot
import shap
import math

from xgboost import to_graphviz
from xgboost import XGBRegressor
from xgboost import plot_importance
from xgboost import plot_tree

from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def LoadCSV():
    df = pd.read_csv(CsvName)
    
    #--Removing Null Values -- 
    
    
    '''
    df = df.iloc[:-1, :]
    df.drop(range(1,13), axis=0,inplace=True)
    df.drop(columns=coldrop,inplace=True)
    nullseries = df.isnull().sum() 
    print(nullseries[nullseries > 0])
    '''
    
    #ToDo: Fill NA with closet value or KNN
    #ToDo: Transform Data based on transform codes or just do Y/Y
    
    #Dropping Transform Row  and resetting index
    df = df.iloc[1:-1, :]
    
    df['sasdate'] = pd.to_datetime(df['sasdate'])
    df.reset_index(drop=True,inplace=True)
    
    return df



    
def XgBoostModel(df,Forecast):
    xdf = df.copy(deep=True) 
    

    #--Splitting Dep and Indep variables --
    ydf = xdf[ivars]
    xdf.drop(ivarslist, axis=1,inplace=True)
    
    #--Removing Date Column--
    xdf.drop(columns='sasdate',inplace=True)
    ydf.drop(columns='sasdate',inplace=True)
    
    if Forecast > 0:
        #Shift CPI Data
        ydf[ivarslist[0]]= ydf[ivarslist[0]].shift(-Forecast)
        
        #Dropping NaN rows from forecast shift
        #ydf = ydf.iloc[:-Forecast, :]
        #xdf = xdf.iloc[:-Forecast, :]
    
    
    '''
    print (ydf.tail(3))
    #print (ydf.head(3))
    print(xdf.shape)
    print(ydf.shape)
    print('   ')
    '''
    
   
    xdflen = len(xdf)-Forecast
 
    
    X_train, X_test, y_train, y_test = train_test_split(xdf.iloc[:xdflen, :], ydf.iloc[:xdflen, :], test_size=0.33, random_state=0, shuffle=True)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    model = XGBRegressor() 
    #Fitting Model 
    model.fit(X_train, y_train, eval_metric="error",verbose=False,eval_set=eval_set)
        
    #Calculating Feature Importance and adding to Feature DF
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_train)
    

    #Creating Feature Importance DF
    vals= np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(xdf.columns,vals)),columns=['Feature','Importance Values'])
    feature_importance.sort_values(by=['Importance Values'],ascending=False,inplace=True)
    feature_importance.reset_index(drop=True,inplace=True)
    
    
    FeatureDF.loc[:,(f'{Forecast}-Month Horizon', 'Feature')] = feature_importance.loc[:,'Feature']
    FeatureDF.loc[:,(f'{Forecast}-Month Horizon', 'Importance Values')] = feature_importance.loc[:,'Importance Values']
    
    
    shap.summary_plot(shap_values, X_train,show=True)
    #shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0],feature_names = xdf.columns)
    #shap.dependence_plot("rank(0)", shap_values, X_train, feature_names=xdf.columns)
    #TO DO: Save plots
    
    '''
    format = 'png'
    image  = to_graphviz(model)
    #Set a different dpi (work only if format == 'png')
    image.graph_attr = {'dpi':'400'}

    image.render('filename', format = format)
    '''

    
    y_pred = model.predict(xdf)
    predictions = [round(value,4) for value in y_pred]
    for x in range(1,Forecast+1):
        predictions.insert(0,np.nan)
    predictions = pd.Series(predictions, name=f'{Forecast}-Month Prediction')


    #Adding in new dates to fill in predictions
    ResultsDF.loc[len(ResultsDF),'sasdate'] =  ResultsDF.loc[len(ResultsDF)-1,'sasdate'] + pd.DateOffset(months=1)
    ResultsDF[f'{Forecast}-Month Prediction'] = predictions
    
    #Filling Errors DF
    ErrorsDF[f'{Forecast}-Month Error'] = ResultsDF[f'{Forecast}-Month Prediction'] - ErrorsDF[ivarslist[0]]
    

    #Calculate RMSE (being further away is more important)
    MSE = mean_squared_error(ResultsDF.loc[Forecast:ResultsDFlen-1,ivarslist[0]], ResultsDF.loc[Forecast:ResultsDFlen-1,f'{Forecast}-Month Prediction'])
    RMSE = round(math.sqrt(MSE),4)
    print(f'{Forecast}-Month Forecast RMSE: {RMSE}')
    
    
    ResultsDF[f'{Forecast}-Month Prediction'] = round(ResultsDF[f'{Forecast}-Month Prediction'],3)

    
    
    return 
    
def plotydf():
    fig = make_subplots(rows=2,  cols=2, shared_xaxes=False, 
                            row_heights=[0.50,0.50],
                            vertical_spacing=0.075,
                            horizontal_spacing=0.075,
                            subplot_titles=("Inflation", "N/A", "N/A", "N/A"))
    
    #Inflation charts
    for x in ResultsDF.columns[1:]:
        fig.add_trace(go.Scatter(x=ResultsDF['sasdate'], y=ResultsDF[x], name=x), row=1, col=1)
        

    fig.update_traces(mode="lines", hovertemplate=None)
    
    #Error Scatter Plot
    for y in ErrorsDF.columns[1:]:
        fig.add_trace(go.Scatter(x=ErrorsDF['sasdate'], y=ErrorsDF[y], name=y,mode="markers"), row=2, col=1)
        
    
  
    
    fig.update_xaxes(
        #rangebreaks=[dict(bounds=["sat", "mon"])], #hide weekends
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
                ])
            )
        )
    
    
    
    fig.update_layout(title_text='Inflation Model',
                        xaxis_showticklabels=True,
                        xaxis2_showticklabels=True,
                        title_x=0.5,
                        title_y=0.99,
                        title_font_color="#333",
                        #hovermode='x unified',

                        )
    
    fig.update_xaxes(row=1, col=1, matches='x')
    fig.update_xaxes(row=2, col=1, matches='x')
    fig.update_xaxes(row=1, col=2, matches='x')
    fig.update_xaxes(row=2, col=2, matches='x')
    
    fig.update_yaxes(
        side="right"
        )
    
    
    pio.write_html(fig, file='InflationModel.html', auto_open=False)



# =============== Running Script ==================
if __name__ == '__main__':
    '''
    ---To Do---
    
    - Transform Data
    - Find Most Important Features
    - Find important stats for the model 
    - Research different model parameters?
    
    - Remove Correlated Variables?
    
    '''

    
    #---Script Variables---
    ScriptVars()

    ForecastRange = 0
    
    #creating X and Y DFs
    CsvName = 'current.csv'
    ivars = ['sasdate']
    ivarslist = ['CPIAUCSL']
    ivars = ivars + ivarslist
    coldrop = ['ACOGNO','ANDENOx']
    
    print("\nCreating DFs")
    df = LoadCSV()
    
    
    
    #Creating Feature DF
    FeatureArry = []
    for i in range(0,ForecastRange+1):
        FeatureArry.append((f'{i}-Month Horizon','Feature'))
        FeatureArry.append((f'{i}-Month Horizon','Importance Values'))
    FeatureIndex = pd.MultiIndex.from_tuples(FeatureArry)
    FeatureDF = pd.DataFrame(columns=FeatureIndex)
    
    #Creating Results DF
    ResultsDF = pd.DataFrame()
    ResultsDF = df[ivars]

    ErrorsDF = ResultsDF.copy(deep=True) 
    
    ResultsDFlen = len(ResultsDF)
    print('\n---Running XgBoost Model---')
    for x in range(0,ForecastRange+1):
        XgBoostModel(df,x)
        
    
    ErrorsDF.pop('CPIAUCSL')
    ResultsDF = ResultsDF.iloc[:-1, :]
    
    FeatureDF.to_excel("FeatureDF.xlsx")
    ResultsDF.to_excel("ResultsDF.xlsx")
    
    
    
 
    print(ResultsDF.head(5))
    print('   ')
    print(ResultsDF.tail(5))
    
    print(ResultsDF.shape)
    #print(ErrorsDF.tail(5))
    
    
    
    #print(FeatureDF.head(10))
    #print(FeatureDF.shape)
    #print(ResultsDF.head(5))
    #print (ResultsDF.shape)
    
    
    #Creating Plotly Charts
    print('\nCreating Plotly Charts')
    plotydf()
    '''
    
    Prediction Scatterplot over time?
    
    '''
    
    
    
    
    '''
    html = FeatureDF.to_html()
    # write html to file
    text_file = open("FeatureDF.html", "w")
    text_file.write(html)
    text_file.close()
    
    
    
    
    print("\nRunning Untransformed Feature Importance and Model")
    #UTModel()
    
    print("\nRunning Transformed Feature Importance and Model")
    #TModel()
    
    
    
    #dfinfo(df)
    
    '''