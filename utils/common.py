import os
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
from scipy.stats import boxcox
from scipy.special import inv_boxcox 
from flask import Flask, current_app
import io
import json
import boto3
import numpy as np
import datetime
import jwt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json

def makeFolder():
    """ Make directory if directory is not available
    """
    try:
        if not os.path.exists('uploads'):
            os.makedirs('uploads')
        return 'uploads'
    except OSError as e:
        print(e)

def test_stationarity(dataframe):
    #Perform Dickey-Fuller test:
    dftest = adfuller(dataframe, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    return dfoutput

def acf_array(dataframe, lags):
    lag_acf = acf(dataframe, nlags=lags)
    return lag_acf

def pacf_array(dataframe, lags):
    lag_pacf = pacf(dataframe, nlags=lags, method='ols')
    return lag_pacf

def decompose(dataframe):
    decomposition = seasonal_decompose(dataframe['closing_balance'], period = 31)
    return decomposition

def train_sarimax(dataframe, p, d, q, P, D, Q, s, test, BOXCOX):
    print(dataframe)
    if(BOXCOX == "Yes"):
        dataframe['closing_balance'], lam = boxcox(dataframe['closing_balance'])
    else:
        lam = None
    if(P == 0 and D == 0 and Q == 0 and s == 0):
        model = ARIMA(dataframe, order=(p, d, q))
    else:
        model = SARIMAX(dataframe,  
                    order = (p, d, q),  
                    seasonal_order =(P, D, Q, s)) 
    results_ARIMA = model.fit(disp=-1)
    ARIMA_predict = pd.DataFrame(results_ARIMA.predict(start = test.index[0], end = test.index[-1]), index = test.index)
    print(results_ARIMA.fittedvalues)
    if(BOXCOX == "Yes"):
        results_ARIMA.fittedvalues = inv_boxcox(results_ARIMA.fittedvalues, lam)
        ARIMA_predict = inv_boxcox(ARIMA_predict, lam)
        ARIMA_predict.columns = ['y']
        response = {}
        response['model'] = results_ARIMA
        response['predictions'] = ARIMA_predict
        response['lam'] = lam
        return response
    else:
        ARIMA_predict.columns = ['y']
        response = {}
        response['model'] = results_ARIMA
        response['predictions'] = ARIMA_predict
        response['lam'] = lam
        return response

def train_arima(dataframe, p, d, q, test, BOXCOX):
    print(dataframe)
    if(BOXCOX == "Yes"):
        dataframe['closing_balance'], lam = boxcox(dataframe['closing_balance'])
    else:
        lam = None
    model = ARIMA(dataframe, order=(p, d, q))
    results_ARIMA = model.fit(disp=-1)
    ARIMA_predict = pd.DataFrame(results_ARIMA.predict(start = test.index[0], end = test.index[-1]), index = test.index)
    print(results_ARIMA.fittedvalues)
    if(BOXCOX == "Yes"):
        results_ARIMA.fittedvalues = inv_boxcox(results_ARIMA.fittedvalues, lam)
        ARIMA_predict = inv_boxcox(ARIMA_predict, lam)
        ARIMA_predict.columns = ['y']
        response = {}
        response['model'] = results_ARIMA
        response['predictions'] = ARIMA_predict
        response['lam'] = lam
        return response
    else:
        ARIMA_predict.columns = ['y']
        response = {}
        response['model'] = results_ARIMA
        response['predictions'] = ARIMA_predict
        response['lam'] = lam
        return response

def autoArima(dataframe, seasonal, test, BOXCOX):
    if(BOXCOX == "Yes"):
        dataframe['closing_balance'], lam = boxcox(dataframe['closing_balance'])
    else:
        lam = None
    automodel = auto_arima(dataframe, start_p = 1, start_q = 1, 
    max_p = 3, max_q = 3, m = 12, 
    start_P = 0, seasonal = seasonal, 
    d = None, D = 1, trace = True) 
    if(BOXCOX == "Yes"):
        ARIMA_predict = pd.DataFrame(automodel.predict(n_periods = test.shape[0]), index = test.index)
        ARIMA_predict = inv_boxcox(ARIMA_predict, lam)
        ARIMA_predict.columns = ['y']
        results = {}
        results['model'] = automodel
        results['predictions'] = ARIMA_predict
        results['params'] = automodel.get_params()
        results['lam'] = lam
        return results
    else:
        ARIMA_predict = pd.DataFrame(automodel.predict(n_periods = test.shape[0]), index = test.index)
        ARIMA_predict.columns = ['y']
        results = {}
        results['model'] = automodel
        results['predictions'] = ARIMA_predict
        results['params'] = automodel.get_params()
        results['lam'] = lam
        return results

def forecast_arima(dataframe, p, d, q, startDate, endDate, lam):
    if lam:
        dataframe['closing_balance'], lam = boxcox(dataframe['closing_balance'])
    model = ARIMA(dataframe, order=(p, d, q))
    results_ARIMA = model.fit(disp=-1)
    idx = pd.date_range(startDate, endDate)
    ARIMA_predict = pd.DataFrame(results_ARIMA.predict(start = pd.to_datetime(startDate).to_period("D"), end = pd.to_datetime(endDate).to_period("D")), index = idx)
    if lam:
        ARIMA_predict = inv_boxcox(ARIMA_predict, lam)
    ARIMA_predict.columns = ['y']
    ARIMA_predict['x'] = ARIMA_predict.index
    return ARIMA_predict

def forecast_sarimax(dataframe, p, d, q, P, D, Q, s, startDate, endDate, lam):
    if lam:
        dataframe['closing_balance'], lam = boxcox(dataframe['closing_balance'])    
    if(P == 0 and D == 0 and Q == 0 and s == 0):
        model = ARIMA(dataframe, order=(p, d, q))
    else:
        model = SARIMAX(dataframe,  
                    order = (p, d, q),  
                    seasonal_order =(P, D, Q, s)) 
    results_ARIMA = model.fit(disp=-1)
    idx = pd.date_range(startDate, endDate)
    ARIMA_predict = pd.DataFrame(results_ARIMA.predict(start = pd.to_datetime(startDate).to_period("D"), end = pd.to_datetime(endDate).to_period("D")), index = idx)
    if lam:
        ARIMA_predict = inv_boxcox(ARIMA_predict, lam)
    ARIMA_predict.columns = ['y']
    ARIMA_predict['x'] = ARIMA_predict.index
    return ARIMA_predict

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def upload_to_aws(dataframe, s3_file_name):
    try:
        s3 = boto3.client('s3', aws_access_key_id  = current_app.config.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key = current_app.config.get('AWS_SECRET_ACCESS_KEY'))
        csv_buf = io.StringIO()
        dataframe.to_csv(csv_buf, header=True, index=False)
        csv_buf.seek(0)
        s3.put_object(Bucket = current_app.config.get('BUCKET_NAME'), Body=csv_buf.getvalue(), 
                    Key = s3_file_name)
        print("Upload Successful")
        return True
    except Exception as Error:
        print(Error)
        return False

def download_from_aws(s3_file_name):
    try:
        s3 = boto3.client('s3', aws_access_key_id  = current_app.config.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key = current_app.config.get('AWS_SECRET_ACCESS_KEY'))
        obj = s3.get_object(Bucket = current_app.config.get('BUCKET_NAME'), Key = s3_file_name)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        return df
    except Exception as Error:
        print(Error)
        print("Aws file download error")
        return False

def upload_model_aws(filename, jsonData):
    try:
        s3 = boto3.client('s3', aws_access_key_id  = current_app.config.get('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key = current_app.config.get('AWS_SECRET_ACCESS_KEY'))
        s3.put_object(Bucket = current_app.config.get('BUCKET_NAME'), Body=jsonData, 
                    Key = filename)
        print("Successfully uploaded model")
        return True
    except Exception as Error:
        print(Error)
        return False

def download_model_aws(model_name):
    print(model_name)
    try:
        s3 = boto3.client('s3', aws_access_key_id  = current_app.config.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key = current_app.config.get('AWS_SECRET_ACCESS_KEY'))
        obj = s3.get_object(Bucket = current_app.config.get('BUCKET_NAME'), Key = model_name)
        model_json_string = obj['Body'].read().decode()
        print(model_json_string)
        return model_json_string
    except Exception as Error:
        print(Error)
        print("Aws file download error")
        return False

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    # print(dataX)
    # print(dataY)
    return np.array(dataX), np.array(dataY)

def lstm(dataframe, testDataframe, filename, LOOK_BACK, BATCH_SIZE, EPOCHS):
    ### SET SEED ###
    np.random.seed(7)

    dataset = dataframe.values
    dataset = dataset.astype('float32')

    testDataset = testDataframe.values
    testDataset = testDataset.astype('float32')

    # normalize the train dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    testDataset = scaler.fit_transform(testDataset)

    # reshape into X=t and Y=t+1
    look_back = LOOK_BACK
    trainX, trainY = create_dataset(dataset, look_back)
    testX, testY = create_dataset(testDataset, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # create and fit the LSTM network
    batch_size = BATCH_SIZE
    model = Sequential()
    model.add(LSTM(5, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(5, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
    model.add(LSTM(5, batch_input_shape=(batch_size, look_back, 1), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(EPOCHS):
        model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
        model.reset_states()

    # make predictions
    trainPredict = model.predict(trainX, batch_size=batch_size)
    model.reset_states()
    testPredict = model.predict(testX, batch_size=batch_size)

    print(testX)
    print("~~~~~~~~~~~~~~~~~~~~~")
    print(testPredict)

    # prediction_list = (trainX)[-look_back:]

    # for _ in range(31 - 1):
    #     x = prediction_list[-look_back:]
    #     print(x)
    #     x = x.reshape((1, look_back, 1))
    #     out = model.predict(x, batch_size=batch_size)[0][0]
    #     prediction_list = np.append(prediction_list, out)
    # prediction_list = prediction_list[look_back-1:]

    # prediction_list = scaler.inverse_transform([prediction_list])

    # print(prediction_list)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    jsonData = model.to_json()
    upload_model_aws(filename+".json", jsonData)
    return trainPredict.flatten(), testPredict.flatten()

def predict(dataframe, startDate, num_prediction, look_back, batch_size, filename):
    print(batch_size)
    dataset = dataframe['closing_balance'].values
    dataset = dataset.astype('float32')

    dataset = dataset.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    model_json_string = download_model_aws(filename+".json")
    model = model_from_json(model_json_string)
    prediction_list = dataset[-look_back:]
    
    for _ in range(num_prediction - 1):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        # print(x)
        # print(model.predict(x, batch_size=batch_size))
        out = model.predict(x, batch_size=batch_size)[0][0]
        prediction_list = np.append(prediction_list, out)
        print(prediction_list)
    prediction_list = prediction_list[look_back-1:]

    prediction_list = scaler.inverse_transform([prediction_list])

    prediction_dates = pd.date_range(startDate, periods=num_prediction).tolist()

    results = pd.DataFrame(prediction_list.flatten())
    results.columns = ['y']
    results['x'] = prediction_dates
    results['x'] = pd.to_datetime(results['x'], format='%Y-%m-%d')
        
    print(prediction_dates)
    print(prediction_list)
    
    return results
    



 



    


