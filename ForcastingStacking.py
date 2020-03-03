# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 19:08:12 2020

@author: gs52078
"""


#Stacking of Different Model

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:35:49 2020

@author: gs52078
"""


#XGBoost for Forcasting

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('fivethirtyeight')


import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

from sklearn.linear_model import LinearRegression

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    #print(df['dayofyear'])
    df['dayofmonth'] = df['date'].dt.day
    #print(df['dayofmonth'])
    df['weekofyear'] = df['date'].dt.weekofyear       
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]#'hour',
    if label:
        y = df[label]
        return X, y
    return X

def XGBoostModel(X_train, y_train,X_test, y_test):
    reg = xgb.XGBRegressor(n_estimators=1000)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
           verbose=True)
    #_ = plot_importance(reg, height=0.9)
    return reg

def MLPModel(X_train, y_train,X_test, y_test):
    epochs = 100
    batch = 128
    lr = 0.0003
    adam = optimizers.Adam(lr)
    
    model_mlp = Sequential()
    model_mlp.add(Dense(100, activation='relu', input_dim=X_train.shape[1]))
    model_mlp.add(Dense(1))
    model_mlp.compile(loss='mse', optimizer=adam)
    model_mlp.summary()
    
    mlp_history = model_mlp.fit(X_train.values, y_train, validation_data=(X_test.values, y_test), epochs=epochs, verbose=2)
    return model_mlp

def CNNLSTMModel(X_train_series, Y_train,X_valid_series, Y_valid):
    epochs = 100
    batch = 128
    lr = 0.0003
    adam = optimizers.Adam(lr)
    
    subsequences = 2
    timesteps = X_train_series.shape[1]//subsequences
    X_train_series_sub = X_train_series.reshape((X_train_series.shape[0], subsequences, timesteps, 1))
    X_valid_series_sub = X_valid_series.reshape((X_valid_series.shape[0], subsequences, timesteps, 1))
    print('Train set shape', X_train_series_sub.shape)
    print('Validation set shape', X_valid_series_sub.shape)
    model_cnn_lstm = Sequential()
    model_cnn_lstm.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, X_train_series_sub.shape[2], X_train_series_sub.shape[3])))
    model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model_cnn_lstm.add(TimeDistributed(Flatten()))
    model_cnn_lstm.add(LSTM(50, activation='relu'))
    model_cnn_lstm.add(Dense(1))
    model_cnn_lstm.compile(loss='mse', optimizer=adam)
    cnn_lstm_history = model_cnn_lstm.fit(X_train_series_sub, Y_train, validation_data=(X_valid_series_sub, Y_valid), epochs=epochs, verbose=2)
    return model_cnn_lstm

def XGBoostPrediction(model, X_test,demandForcastingData_test,demandForcastingData_train ):
    demandForcastingData_test['Count_Prediction1'] = model.predict(X_test)
    demandForcastingData_all = pd.concat([demandForcastingData_test, demandForcastingData_train], sort=False)

    #_ = demandForcastingData_all[['Count','Count_Prediction']].plot(figsize=(15, 5))
    return demandForcastingData_test

def MLPPrediction(model, X_test,demandForcastingData_test,demandForcastingData_train ):
    demandForcastingData_test['Count_Prediction2'] = model.predict(X_test)
    demandForcastingData_all = pd.concat([demandForcastingData_test, demandForcastingData_train], sort=False)

    #_ = demandForcastingData_all[['Count','Count_Prediction']].plot(figsize=(15, 5))
    return demandForcastingData_test
def LSTMCNNPrediction(model, X_test,demandForcastingData_test,demandForcastingData_train ):
    X_test_series = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    subsequences = 2
    timesteps = X_test_series.shape[1]//subsequences
    X_test_series_sub = X_test_series.reshape((X_test_series.shape[0], subsequences, timesteps, 1))
    demandForcastingData_test['Count_Prediction3'] = model.predict(X_test_series_sub)
    demandForcastingData_all = pd.concat([demandForcastingData_test, demandForcastingData_train], sort=False)
    #print(demandForcastingData_test['Count_Prediction'])
    #_ = demandForcastingData_all[['Count','Count_Prediction']].plot(figsize=(15, 5))
    return demandForcastingData_test


def modelEvaluation(y_test, y_pred):    
    error_margin = 0.3
    relative_errors = abs((y_test - y_pred)/y_pred.mean())
    correct_preds = (relative_errors <= error_margin)
    print(correct_preds)
    
    result_dict = dict(correct_preds.value_counts())
    print(result_dict)
    #print(result_dict[False])
    if False in result_dict:
        accuracy = 100*result_dict[True]/(result_dict[True]+result_dict[False])
    else:
        accuracy = 100*result_dict[True]/(result_dict[True])
    print('accuracy: %.2f %%'% (accuracy))

def prophetModelandPrediction(demandForcastingData_train,demandForcastingData_test,holiday_df):
    cal = calendar()
    #train_holidays = cal.holidays(start=demandForcastingData_train.index.min(),end=demandForcastingData_train.index.max())
    #test_holidays = cal.holidays(start=demandForcastingData_test.index.min(),end=demandForcastingData_test.index.max())
    holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])
    demandForcastingData_train.reset_index().rename(columns={'ArrivalDate':'ds','Count':'y'}).head()
    model = Prophet(holidays=holiday_df)
    model.fit(demandForcastingData_train.reset_index().rename(columns={'ArrivalDate':'ds','Count':'y'}))
    demandForcastingData_test_fcst = model.predict(df=demandForcastingData_test.reset_index().rename(columns={'ArrivalDate':'ds'}))
    demandForcastingData_test['Count_Prediction4'] = demandForcastingData_test_fcst.yhat.values
    #print(predProphet)
    return demandForcastingData_test
    

def prophetMethod(demandForcastingData,demandForcastingData_train,demandForcastingData_test):
    cal = calendar()
    demandForcastingData['date'] = demandForcastingData.index.date
    demandForcastingData['is_holiday'] = demandForcastingData.date.isin([d.date() for d in cal.holidays()])
    holiday_df = demandForcastingData.loc[demandForcastingData['is_holiday']].reset_index().rename(columns={'ArrivalDate':'ds'})
    holiday_df['holiday'] = 'Holiday'
    holiday_df = holiday_df.drop(['Count','date','is_holiday'], axis=1)
    holiday_df['ds'] = pd.to_datetime(holiday_df['ds'])
    
    demandForcastingData_test=prophetModelandPrediction(demandForcastingData_train,demandForcastingData_test,holiday_df)
    #modelEvaluation(y_test=demandForcastingData_test['Count'],y_pred=demandForcastingData_test['Count_Prediction'])
    return demandForcastingData_test

def xgboostMethod(X_train, y_train,X_test, y_test,demandForcastingData_train,demandForcastingData_test):
    model=XGBoostModel(X_train, y_train,X_test, y_test)
    demandForcastingData_test=XGBoostPrediction(model, X_test,demandForcastingData_test,demandForcastingData_train )
    #modelEvaluation(y_test=demandForcastingData_test['Count'],y_pred=demandForcastingData_test['Count_Prediction'])
    return demandForcastingData_test

def MLPMethod(X_train, y_train,X_test, y_test,demandForcastingData_train,demandForcastingData_test):
    model=MLPModel(X_train, y_train,X_test, y_test)
    demandForcastingData_test=MLPPrediction(model, X_test,demandForcastingData_test,demandForcastingData_train )
    #modelEvaluation(y_test=demandForcastingData_test['Count'],y_pred=demandForcastingData_test['Count_Prediction'])
    return demandForcastingData_test

def CNNLSTMMethod(X_train, y_train,X_test, y_test,demandForcastingData_train,demandForcastingData_test):
    X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_series = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    print('Train set shape', X_train_series.shape)
    print('Validation set shape', X_test_series.shape)

    modelDL= CNNLSTMModel(X_train_series, y_train,X_test_series, y_test)
    demandForcastingData_test=LSTMCNNPrediction(modelDL, X_test,demandForcastingData_test,demandForcastingData_train )
    #modelEvaluation(y_test=demandForcastingData_test['Count'],y_pred=demandForcastingData_test['Count_Prediction'])
    return demandForcastingData_test

# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(models, predict_list, row):
	stacked_row = list()
	for i in range(len(models)):
		prediction = predict_list[i](models[i], row)
		stacked_row.append(prediction)
	stacked_row.append(row[-1])
	return row[0:len(row)-1] + stacked_row
 
# Stacked Generalization Algorithm
def stacking(train, test):
	model_list = [xgboost, MLPModel,CNNLSTM,ProphetModel]
	predict_list = [xgboost_pred, MLPModel_pred,CNNLSTM_pred,ProphetModel_pred]
	models = list()
	for i in range(len(model_list)):
		model = model_list[i](train)
		models.append(model)
	stacked_dataset = list()
	for row in train:
		stacked_row = to_stacked_row(models, predict_list, row)
		stacked_dataset.append(stacked_row)
	stacked_model = logistic_regression_model(stacked_dataset)
	predictions = list()
	for row in test:
		stacked_row = to_stacked_row(models, predict_list, row)
		stacked_dataset.append(stacked_row)
		prediction = logistic_regression_predict(stacked_model, stacked_row)
		prediction = round(prediction)
		predictions.append(prediction)
	return predictions
def main():
    demandForcastingData = pd.read_csv('D:/Practice/AmericasBestValue.csv', index_col=[0], parse_dates=[0])

    color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
    #_ = demandForcastingData.plot(style='.', figsize=(15,5), color=color_pal[0], title='demand ForcastingData')
    
    split_date = '2019-09-02'
    demandForcastingData_train = demandForcastingData.loc[demandForcastingData.index <= split_date].copy()
    demandForcastingData_test = demandForcastingData.loc[demandForcastingData.index > split_date].copy()
    #-----------------Plot
    '''_ = demandForcastingData_test \
        .rename(columns={'Count': 'TEST SET'}) \
        .join(demandForcastingData_train.rename(columns={'Count': 'TRAINING SET'}), how='outer') \
        .plot(figsize=(15,5), title='demand Forcasting Data', style='.')'''
    X_train, y_train = create_features(demandForcastingData_train, label='Count')
    X_test, y_test = create_features(demandForcastingData_test, label='Count')
    #forcast Visualization
    '''f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = demandForcastingData_all[['Count_Prediction','Count']].plot(ax=ax,
                                                  style=['-','.'])
    ax.set_xbound(lower='2019-11-01', upper='2019-12-31')
    ax.set_ylim(0, 60)
    plot = plt.suptitle('January 2020 Forecast vs Actuals')'''
    #Model 1: XGBoost
    modelXGB=xgboostMethod(X_train, y_train,X_test, y_test,demandForcastingData_train,demandForcastingData_test)
    
    #Model 2: MLP
    modelMLP=MLPMethod(X_train, y_train,X_test, y_test,demandForcastingData_train,demandForcastingData_test)
    
    #Model 3: CNNLSTM
    modelCNNLSTM=CNNLSTMMethod(X_train, y_train,X_test, y_test,demandForcastingData_train,demandForcastingData_test)
    
    #Model 4: ProphetModel
    modelPRO=prophetMethod(demandForcastingData,demandForcastingData_train,demandForcastingData_test)
    #print(modelXGB)
    #n_folds = 3
    #scores = evaluate_algorithm(dataset, stacking, n_folds)
    stackedOutModel = np.column_stack((modelXGB['Count_Prediction1'], modelMLP['Count_Prediction2'],modelCNNLSTM['Count_Prediction3'], modelPRO['Count_Prediction4']))
    print(stackedOutModel)
    #regressor = LinearRegression()
    #regressor.fit(stackedOutModel, y_test)
    #-----
    xgbreg = xgb.XGBRegressor(n_estimators=100)
    xgbreg.fit(stackedOutModel, y_test,
            eval_set=[(stackedOutModel, y_test), (stackedOutModel, y_test)],
            early_stopping_rounds=50,
           verbose=True)
    demandForcastingData_test['Count_PredictionFinal'] = xgbreg.predict(stackedOutModel)
    modelEvaluation(y_test=demandForcastingData_test['Count'],y_pred=demandForcastingData_test['Count_PredictionFinal'])
    #-----
    #print(demandForcastingData_test['Count'])
    #print(demandForcastingData_test['Count_PredictionFinal'])
    #y_pred = regressor.predict(stackedOutModel)
    #modelEvaluation(y_test=demandForcastingData_test['Count'],y_pred=y_pred)
    
if __name__ == '__main__':
    main()
