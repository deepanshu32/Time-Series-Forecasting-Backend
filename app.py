from flask import Flask, request, redirect
from flask_cors import CORS, cross_origin
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import os
import pandas as pd
import numpy as np
import datetime
from calendar import monthrange
from utils import common
import uuid
import re
import json
from flask_pymongo import PyMongo
import bcrypt
from bson import ObjectId

app = Flask(__name__)
app.config.from_object('config.BaseConfig')
cors = CORS(app)
common.makeFolder()
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print("---------", app.config['UPLOAD_FOLDER'])
jwt = JWTManager(app)

if app.config['MONGO_DBNAME'] and app.config['MONGO_URI']:
    mongo = PyMongo(app)

@app.route('/')
@cross_origin()
def index():
    try:
        """ Default route 
        """
        return "SME Credit"
    except Exception as Error:
        print("Index Route-->", Error)
        return "Bad Request", 400

@app.route('/profile')
@jwt_required
def profile():
    try:
        current_user = get_jwt_identity()
        print(current_user)
        response = {}
        users = mongo.db.users
        user = users.find_one({'_id' : ObjectId(current_user)})
        user['_id'] = str(user['_id'])
        user['password'] = None
        response['user'] = user
        return json.dumps(response)
    except Exception as Error:
        print(Error)
        return "Internal Server Error", 500

@app.route('/login', methods=['POST'])
def login():
    users = mongo.db.users
    login_user = users.find_one({'email' : request.form['email']})
    login_user['_id'] = str(login_user['_id'])

    if login_user:
        if bcrypt.hashpw(request.form['password'].encode('utf-8'), login_user['password']) == login_user['password']:
            expires = datetime.timedelta(days=1)
            auth_token = create_access_token(identity=login_user['_id'], expires_delta=expires)
            print(auth_token)
            login_user['password'] = None
            response = {}
            response['user'] = login_user
            response['token'] = auth_token
            print(response)
            return json.dumps(response)

    return "Invalid email/password combination", 401       

@app.route('/register', methods=['POST'])
def register():
    print(request.headers)
    if request.method == 'POST':
        users = mongo.db.users
        existing_user = users.find_one({'email' : request.form['email']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())
            new_user = users.insert_one({'email' : request.form['email'], 'password' : hashpass})
            user = users.find_one({'_id' : new_user.inserted_id})
            user['_id'] = str(user['_id'])
            user['password'] = None
            expires = datetime.timedelta(days=1)
            auth_token = create_access_token(identity=user['_id'], expires_delta=expires)
            print(auth_token)
            response = {}
            response['user'] = user
            response['token'] = auth_token
            return json.dumps(response)

        return "Email already registered!", 422

@app.route('/upload', methods=['POST'])
@cross_origin()
def upload():
    """ Route for upload bank statements 
    """
    try:
        if request.method == 'POST':
            if request.files and request.files['statements']:
                statements = request.files.getlist('statements')
                li = []
                for file in statements:
                    filename = str(uuid.uuid4())+".csv"
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    df = pd.read_csv("./uploads/"+filename, index_col=None, header=0, error_bad_lines=False)
                    li.append(df)
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                df = pd.concat(li, axis=0, ignore_index=True)
                RE_FILTER_ALL = re.compile('balance|date', re.IGNORECASE)
                RE_FILTER_BALANCE = re.compile('(\w*)balance(\w*)', re.IGNORECASE)
                RE_FILTER_DATE = re.compile('(\w*)date(\w*)', re.IGNORECASE)
                df = df.filter(regex=RE_FILTER_ALL)
                df.columns = df.columns.str.replace(RE_FILTER_BALANCE, 'closing_balance', regex=True) 
                df.columns = df.columns.str.replace(RE_FILTER_DATE, 'indexDate', regex=True) 
                df['closing_balance'] = df['closing_balance'].str.replace(',', '').astype(float)
                df['indexDate'] =  pd.to_datetime(df['indexDate'], format='%Y-%m-%d')
                df.sort_values(by=['indexDate'], inplace=True, ascending=True)
                if request.form and ('fillMissing' in request.form or 'uniform' in request.form):
                    startDate = pd.to_datetime(str(df['indexDate'].values[0]))
                    startDate = startDate.strftime('%Y-%m-%d')
                    startDate = datetime.datetime.strptime(startDate, "%Y-%m-%d")
                    startDate = datetime.datetime(startDate.year, startDate.month, 1)
                    endDate = pd.to_datetime(str(df['indexDate'].values[-1]))
                    endDate = endDate.strftime('%Y-%m-%d')
                    endDate = datetime.datetime.strptime(endDate, "%Y-%m-%d")
                    endDay = monthrange(endDate.year, endDate.month)[1]
                    endDate = datetime.datetime(endDate.year, endDate.month, endDay)
                    idx = pd.date_range(startDate, endDate)
                    df.index = df['indexDate']
                    df = df.reindex(idx)
                    df['indexDate'] = idx
                    df = df.fillna(method='ffill')
                    df = df.fillna(method='backfill')
                if request.form and 'uniform' in request.form:
                    last_days_of_month_dataframe = df.groupby(df.index.month).tail(1) 
                    for index, row in last_days_of_month_dataframe.iterrows():
                        if(index.day == 28):
                            df = df.append(pd.DataFrame({'closing_balance': row["closing_balance"], 'indexDate': row["indexDate"]}, index=[index]))
                            df = df.append(pd.DataFrame({'closing_balance': row["closing_balance"], 'indexDate': row["indexDate"]}, index=[index]))
                            df = df.append(pd.DataFrame({'closing_balance': row["closing_balance"], 'indexDate': row["indexDate"]}, index=[index]))
                        elif(index.day == 29):
                            df = df.append(pd.DataFrame({'closing_balance': row["closing_balance"], 'indexDate': row["indexDate"]}, index=[index]))
                            df = df.append(pd.DataFrame({'closing_balance': row["closing_balance"], 'indexDate': row["indexDate"]}, index=[index]))
                        elif(index.day == 30):
                            df = df.append(pd.DataFrame({'closing_balance': row["closing_balance"], 'indexDate': row["indexDate"]}, index=[index]))
                    df.sort_index(inplace=True, ascending=True)
                filename = str(uuid.uuid4())
                train = df.iloc[0 : int(len(df.index)*(1 - (int(request.form['split'])/100)))]
                test = df.iloc[int(len(df.index)*(1 - (int(request.form['split'])/100))) : len(df.index)]
                upload_train = common.upload_to_aws(train, filename+".csv")
                upload_test = common.upload_to_aws(test, filename+"_test.csv")
                df.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
                response = {}
                response['success'] = True
                response['filename'] = filename+".csv"
                response['data'] = json.loads(df.to_json(orient='records'))
                response['split'] = int(request.form['split'])
                return response
        else:
            return "Not Found", 404
    except Exception as Error:
        print("Upload Error-->", Error)
        return "Bad Request", 400

@app.route('/plotInitial', methods=['POST'])
@cross_origin()
def plotInitial():
    try:
        if request.form and 'filename' in request.form:
            train = common.download_from_aws(request.form['filename'])
            train['indexDate'] =  pd.to_datetime(train['indexDate'], format='%Y-%m-%d')
            train.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
            response = {}
            response['success'] = True
            response['data'] = json.loads(train.to_json(orient='records'))
            return response
    except Exception as Error:
        print("Plot Initial Error-->", Error)
        return "Bad Request", 400

@app.route('/stationarity', methods=['POST'])
@cross_origin()
def stationarity():
    try:
        if request.form and 'filename' in request.form:
            train = common.download_from_aws(request.form['filename'])
            train['indexDate'] =  pd.to_datetime(train['indexDate'], format='%Y-%m-%d')
            train.index = train['indexDate']
            trainCopy = train.copy()
            del trainCopy['indexDate']
            results = common.test_stationarity(trainCopy)
            results = json.loads(results.to_frame().to_json())
            trainCopy1 = train.copy()
            trainCopy2 = train.copy()
            trainCopy1["y"] =  train['closing_balance'].rolling(window=15).mean()
            trainCopy2["y"] = train['closing_balance'].rolling(window=15).std()
            del trainCopy1['closing_balance']
            del trainCopy2['closing_balance']
            train.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
            trainCopy1.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
            trainCopy2.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
            response = {}
            response["results"] = results
            response["mean"] = json.loads(trainCopy1.to_json(orient='records'))
            response["std"] = json.loads(trainCopy2.to_json(orient='records'))
            response["data"] = json.loads(train.to_json(orient='records'))
            return response
        else:
            return "Bad Request", 400
    except Exception as Error:
        print("Stationary Test Error-->", Error)
        return "Bad Request", 400

@app.route('/average', methods=['POST'])
@cross_origin()
def average():
    try:
        print(request.form)
        if request.form and ('filename' in request.form and 'testFilename' in request.form and 'split' in request.form and 'day' in request.form):
            if(request.form['type'] == "before"):
                train = common.download_from_aws(request.form['filename'])
                test = common.download_from_aws(os.path.splitext(request.form['testFilename'])[0]+"_test.csv")
                dataset = pd.concat([train, test])
                dataset = dataset.reset_index(drop=True)
                dataset['closing_balance'] = dataset['closing_balance'].rolling(window=int(request.form['day'])).mean()
                dataset = dataset.dropna()
                train = dataset.iloc[0 : int(len(dataset.index)*(1 - (int(request.form['split'])/100)))]
                test = dataset.iloc[int(len(dataset.index)*(1 - (int(request.form['split'])/100))) : len(dataset.index)]
                filename = os.path.splitext(request.form['filename'])[0]
                filename = filename+"_"+request.form['day']+"_day_avg"
                upload_avg_train = common.upload_to_aws(train, filename+".csv")
                upload_avg_test = common.upload_to_aws(test, filename+"_test.csv")
                dataset['indexDate'] =  pd.to_datetime(dataset['indexDate'], format='%Y-%m-%d')
                dataset.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
                response = {}
                response['filename'] = filename+".csv"
                response['data'] = json.loads(dataset.to_json(orient='records'))
                response['days'] = request.form['day']
                return response
            elif(request.form['type'] == "after"):
                train = common.download_from_aws(request.form['filename'])
                test = common.download_from_aws(os.path.splitext(request.form['testFilename'])[0]+"_test.csv")
                dataset = pd.concat([train, test])
                dataset = dataset.reset_index(drop=True)
                dataset['closing_balance'] = dataset['closing_balance'].rolling(window=int(request.form['day'])).mean()
                dataset = dataset.dropna()
                dataset.sort_index(inplace=True, ascending=False)
                dataset['closing_balance'] = dataset['closing_balance'].rolling(window=int(request.form['day'])).mean()
                dataset = dataset.dropna()
                dataset.sort_index(inplace=True, ascending=True)
                train = dataset.iloc[0 : int(len(dataset.index)*(1 - (int(request.form['split'])/100)))]
                test = dataset.iloc[int(len(dataset.index)*(1 - (int(request.form['split'])/100))) : len(dataset.index)]
                filename = os.path.splitext(request.form['filename'])[0]
                filename = filename+"_"+request.form['day']+"_day_avg"
                upload_avg_train = common.upload_to_aws(train, filename+".csv")
                upload_avg_test = common.upload_to_aws(test, filename+"_test.csv")
                dataset['indexDate'] =  pd.to_datetime(dataset['indexDate'], format='%Y-%m-%d')
                dataset.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
                response = {}
                response['filename'] = filename+".csv"
                response['data'] = json.loads(dataset.to_json(orient='records'))
                response['days'] = request.form['day']
                return response
            else:
                return "Bad Request", 400
        else:
            return "Bad Request", 400
    except Exception as Error:
        print("Stationary Test Error-->", Error)
        return "Bad Request", 400

@app.route('/plotAverage', methods=['POST'])
@cross_origin()
def plotAverage():
    try:
        if request.form and 'filename' in request.form:
            train = common.download_from_aws(request.form['filename'])
            train['indexDate'] =  pd.to_datetime(train['indexDate'], format='%Y-%m-%d')
            train.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
            response = {}
            response['success'] = True
            response['data'] = json.loads(train.to_json(orient='records'))
            return response
    except Exception as Error:
        print("Stationary Test Error-->", Error)
        return "Bad Request", 400

@app.route('/corelation', methods=['POST'])
@cross_origin()
def corelation():
    try:
        if request.form and 'filename' in request.form:
            train = common.download_from_aws(request.form['filename'])
            train.index = train['indexDate']
            trainCopy = train.copy()
            acf = common.acf_array(trainCopy['closing_balance'], int(request.form['lags']))
            pacf = common.pacf_array(trainCopy['closing_balance'], int(request.form['lags']))
            response = {}
            response['acf'] = acf.tolist()
            response['pacf'] = pacf.tolist()
            response['size'] = len(train.index)
            return response
        else:
            return "Bad Request", 400
    except Exception as Error:
        print("Corelation Error-->", Error)
        return "Bad Request", 400

@app.route('/decompose', methods=['POST'])
@cross_origin()
def decompose():
    try:
        if request.form and ('filename' in request.form and 'period' in request.form):
            train = common.download_from_aws(request.form['filename'])
            train.index = train['indexDate']
            trainCopy = train.copy()
            decomposition = common.decompose(trainCopy)
            response = {}
            trend = pd.DataFrame(decomposition.trend)
            trend['x'] = pd.to_datetime(trainCopy['indexDate'], format='%Y-%m-%d')
            trend.rename(columns = {'trend':'y'}, inplace = True)
            trend = trend.dropna()
            seasonal = pd.DataFrame(decomposition.seasonal)
            seasonal['x'] = pd.to_datetime(trainCopy['indexDate'], format='%Y-%m-%d')
            seasonal.rename(columns = {'seasonal':'y'}, inplace = True)
            seasonal = seasonal.dropna()
            resid = pd.DataFrame(decomposition.resid)
            resid['x'] = pd.to_datetime(trainCopy['indexDate'], format='%Y-%m-%d')
            resid.rename(columns = {'resid':'y'}, inplace = True)
            resid = resid.dropna()
            response['trend'] = json.loads(trend.to_json(orient='records')) 
            response['seasonal'] = json.loads(seasonal.to_json(orient='records')) 
            response['resid'] = json.loads(resid.to_json(orient='records')) 
            return response
        else:
            return "Bad Request", 400
    except Exception as Error:
        print("Decompose Error-->", Error)
        return "Bad Request", 400

@app.route('/arima', methods=['POST'])
@cross_origin()
def arima():
    try:
        print(request.form)
        if request.form and ('filename' in request.form and 'p' in request.form and 'd' in request.form and 'q' in request.form and request.form['seasonality']):
            train = common.download_from_aws(request.form['filename'])
            test = common.download_from_aws(os.path.splitext(request.form['testFilename'])[0]+"_test.csv")
            train.index = train['indexDate']
            test['indexDate'] = pd.to_datetime(test['indexDate'], format='%Y-%m-%d')
            test.index = test['indexDate']
            test.index = pd.DatetimeIndex(test.index).to_period('D')
            trainCopy = train.copy()
            trainCopy.index = pd.DatetimeIndex(train.index).to_period('D')
            del trainCopy['indexDate']
            if(request.form['seasonality'] == 'true'):
                results = common.train_sarimax(trainCopy, int(request.form["p"]), int(request.form["d"]), int(request.form["q"]), int(request.form["P"]), int(request.form["D"]), int(request.form["Q"]), int(request.form["s"]),test, request.form['boxcox'])
            else:
                results = common.train_arima(trainCopy, int(request.form["p"]), int(request.form["d"]), int(request.form["q"]), test, request.form['boxcox'])
            model = pd.DataFrame(results['model'].fittedvalues)
            predictions = results['predictions']
            predictions['x'] =  predictions.index.to_timestamp('s').strftime('%Y-%m-%d')
            predictions['x'] = pd.to_datetime(predictions['x'], format='%Y-%m-%d')
            model['x'] = model.index
            model.rename( columns={0 :'y'}, inplace=True )
            model['x'] =  model.index.to_timestamp('s').strftime('%Y-%m-%d')
            model['x'] = pd.to_datetime(model['x'], format='%Y-%m-%d')
            train['indexDate'] = pd.to_datetime(train['indexDate'], format='%Y-%m-%d')
            train.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
            test.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
            if(int(request.form["P"]) == 0 and int(request.form["D"]) == 0 and int(request.form["Q"]) == 0 and int(request.form["s"]) == 0):
                mape = common.mean_absolute_percentage_error(train['y'][int(request.form["d"]):], model['y'])
            else:
                mape = common.mean_absolute_percentage_error(train['y'], model['y'])
            mapeValidation = common.mean_absolute_percentage_error(predictions['y'], test['y'])
            response = {}
            response['data'] = json.loads(train.to_json(orient='records'))
            response['results'] = json.loads(model.to_json(orient='records'))
            response['predictions'] = json.loads(predictions.to_json(orient='records'))
            response['test'] = json.loads(test.to_json(orient='records'))
            response['mape'] = mape
            response['mapeValidation'] = mapeValidation
            response['p'] = request.form['p']
            response['q'] = request.form['q']
            response['d'] = request.form['d']
            response['P'] = request.form['P']
            response['Q'] = request.form['Q']
            response['D'] = request.form['D']
            response['s'] = request.form['s']
            response['lam'] = results['lam']
            response['filename'] = request.form['filename']
            return response 
        else:
            return "Bad Request", 400
    except Exception as Error: 
        print("ARIMA Error-->", Error)
        return "Bad Request", 400

@app.route('/autoArima', methods=['POST'])
@cross_origin()
def autoArima():
    try:
        print(request.form)
        if request.form and ('filename' in request.form and 'seasonality' in request.form):
            train = common.download_from_aws(request.form['filename'])
            test = common.download_from_aws(os.path.splitext(request.form['testFilename'])[0]+"_test.csv")
            train['indexDate'] = pd.to_datetime(train['indexDate'], format='%Y-%m-%d')
            train.index = train['indexDate']
            test['indexDate'] = pd.to_datetime(test['indexDate'], format='%Y-%m-%d')
            test.index = test['indexDate']
            trainCopy = train.copy()
            trainCopy.index = pd.DatetimeIndex(train.index).to_period('D')
            del trainCopy['indexDate']
            if(request.form['seasonality'] == 'true'):
                results = common.autoArima(trainCopy, True, test, request.form['boxcox'])
            else:
                results = common.autoArima(trainCopy, False, test, request.form['boxcox'])
            predictions = results['predictions']
            model = results['model']
            predictions['x'] = predictions.index
            train.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
            test.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
            mapeValidation = common.mean_absolute_percentage_error(predictions['y'], test['y'])
            response = {}
            response['data'] = json.loads(train.to_json(orient='records'))
            response['predictions'] = json.loads(predictions.to_json(orient='records'))
            response['test'] = json.loads(test.to_json(orient='records'))
            response['model'] = str(model)
            response['mapeValidation'] = mapeValidation
            response['p'] = results['params']['order'][0]
            response['d'] = results['params']['order'][1]
            response['q'] = results['params']['order'][2]
            response['P'] = results['params']['seasonal_order'][0]
            response['D'] = results['params']['seasonal_order'][1]
            response['Q'] = results['params']['seasonal_order'][2]
            response['s'] = results['params']['seasonal_order'][3]
            response['lam'] = results['lam']
            response['filename'] = request.form['filename']
            return response
        else:
            return "Bad Request", 400
    except Exception as Error: 
        print("ARIMA Error-->", Error)
        return "Bad Request", 400  

@app.route('/lstm', methods=['POST'])
@cross_origin()
def lstm():
    try:
        print(request.form)
        if request.form and 'filename' in request.form and 'testFilename' in request.form and 'look_back' in request.form and 'batch_size' in request.form and 'epochs' in request.form:
            train = common.download_from_aws(request.form['filename'])
            test = common.download_from_aws(os.path.splitext(request.form['testFilename'])[0]+"_test.csv")
            train['indexDate'] = pd.to_datetime(train['indexDate'], format='%Y-%m-%d')
            test['indexDate'] = pd.to_datetime(test['indexDate'], format='%Y-%m-%d')
            trainCopy = train.copy()
            testCopy = test.copy()
            del trainCopy['indexDate']
            del testCopy['indexDate']
            trainPredictions, testPredictions = common.lstm(trainCopy, testCopy, os.path.splitext(request.form['filename'])[0], int(request.form['look_back']), int(request.form['batch_size']), int(request.form['epochs']))
            train.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
            test.rename(columns = {'closing_balance':'y', 'indexDate': 'x'}, inplace = True)
            results = pd.DataFrame(trainPredictions)
            results.columns = ['y']
            predictionsDateTrain = train['x'][int(request.form['look_back'])+1:]
            predictionsDateTrain = predictionsDateTrain.reset_index(drop=True)
            results['x'] = predictionsDateTrain
            predictionsDateTest = test['x'][int(request.form['look_back'])+1:]
            predictionsDateTest = predictionsDateTest.reset_index(drop=True)
            predictions = pd.DataFrame(testPredictions)
            predictions.columns = ['y']
            predictions['x'] = predictionsDateTest
            mape = common.mean_absolute_percentage_error(train['y'][int(request.form['look_back'])+1:], results['y'])
            mapeValidation = common.mean_absolute_percentage_error(test['y'][int(request.form['look_back'])+1:], predictions['y'])
            response = {}
            response['data'] = json.loads(train.to_json(orient='records'))
            response['results'] = json.loads(results.to_json(orient='records'))
            response['predictions'] = json.loads(predictions.to_json(orient='records'))
            response['test'] = json.loads(test.to_json(orient='records'))
            response['mape'] = mape
            response['mapeValidation'] = mapeValidation
            response['model'] = "lstm(look_back="+request.form['look_back']+", batch_size="+request.form['batch_size']+", epochs="+request.form['epochs']+")"
            response['look_back'] = request.form['look_back']
            response['batch_size'] = request.form['batch_size']
            response['epochs'] = request.form['epochs']
            response['filename'] = request.form['filename']
            return response
    except Exception as Error: 
        print("LSTM Error-->", Error)
        return "Bad Request", 400  

@app.route('/forecast', methods=['POST'])
@cross_origin()
def forecastArima(): 
    print(request.form)
    if request.form and ('type' in request.form) and request.form['type'] == 'arima':  
        if request.form and ('filename' in request.form and 'p' in request.form and 'd' in request.form and 'q' in request.form):
            startDate = datetime.datetime.strptime(request.form['startDate'], '%Y-%m-%d')
            endDate = datetime.datetime.strptime(request.form['endDate'], '%Y-%m-%d')
            train = common.download_from_aws(request.form['filename'])
            test = common.download_from_aws(os.path.splitext(request.form['testFilename'])[0]+"_test.csv")
            train['indexDate'] = pd.to_datetime(train['indexDate'], format='%Y-%m-%d')
            train.index = train['indexDate']
            test['indexDate'] = pd.to_datetime(test['indexDate'], format='%Y-%m-%d')
            test.index = test['indexDate']
            trainCopy = train.copy()
            trainCopy.index = pd.DatetimeIndex(train.index).to_period('D')
            del trainCopy['indexDate']
            if request.form['lam'] and request.form['lam'] != "null":
                results = common.forecast_sarimax(trainCopy, int(request.form["p"]), int(request.form["d"]), int(request.form["q"]), int(request.form["P"]), int(request.form["D"]), int(request.form["Q"]), int(request.form["s"]), test.index[0], endDate, request.form["lam"])
            else:
                results = common.forecast_sarimax(trainCopy, int(request.form["p"]), int(request.form["d"]), int(request.form["q"]), int(request.form["P"]), int(request.form["D"]), int(request.form["Q"]), int(request.form["s"]), test.index[0], endDate, None)
            start = results.index.searchsorted(startDate)
            end = results.index.searchsorted(endDate)
            results = results.iloc[start:end]
            response = {}
            response['predictions'] = json.loads(results.to_json(orient='records'))
            return response
    elif request.form and ('type' in request.form) and request.form['type'] == 'auto_arima':
        if 'filename' in request.form:
            startDate = datetime.datetime.strptime(request.form['startDate'], '%Y-%m-%d')
            endDate = datetime.datetime.strptime(request.form['endDate'], '%Y-%m-%d')
            train = common.download_from_aws(request.form['filename'])
            test = common.download_from_aws(os.path.splitext(request.form['testFilename'])[0]+"_test.csv")
            train['indexDate'] = pd.to_datetime(train['indexDate'], format='%Y-%m-%d')
            train.index = train['indexDate']
            test['indexDate'] = pd.to_datetime(test['indexDate'], format='%Y-%m-%d')
            test.index = test['indexDate']
            trainCopy = train.copy()
            trainCopy.index = pd.DatetimeIndex(train.index).to_period('D')
            del trainCopy['indexDate']
            if request.form['lam'] and request.form['lam'] != "null":
                results = common.forecast_sarimax(trainCopy, int(request.form["p"]), int(request.form["d"]), int(request.form["q"]), json.loads((request.form["P"])), int(request.form["D"]), json.loads((request.form["Q"])), int(request.form["s"]), test.index[0], endDate, float(request.form['lam']))
            else:
                results = common.forecast_sarimax(trainCopy, int(request.form["p"]), int(request.form["d"]), int(request.form["q"]), json.loads((request.form["P"])), int(request.form["D"]), json.loads((request.form["Q"])), int(request.form["s"]), test.index[0], endDate, None)
            start = results.index.searchsorted(startDate)
            end = results.index.searchsorted(endDate)
            results = results.iloc[start:end]
            response = {}
            response['predictions'] = json.loads(results.to_json(orient='records'))
            return response
    elif request.form and ('type' in request.form) and request.form['type'] == 'lstm':
        if 'filename' in request.form:
            startDate = datetime.datetime.strptime(request.form['startDate'], '%Y-%m-%d')
            endDate = datetime.datetime.strptime(request.form['endDate'], '%Y-%m-%d')
            train = common.download_from_aws(request.form['filename'])
            test = common.download_from_aws(os.path.splitext(request.form['testFilename'])[0]+"_test.csv")
            dataset = pd.concat([train, test])
            dataset = dataset.reset_index(drop=True)
            lastDate = datetime.datetime.strptime(dataset['indexDate'].values[-1], '%Y-%m-%d')
            train['indexDate'] = pd.to_datetime(train['indexDate'], format='%Y-%m-%d')
            test['indexDate'] = pd.to_datetime(test['indexDate'], format='%Y-%m-%d')
            if((startDate - lastDate).days < 0):
                dropRows = (startDate - lastDate).days - 2
                dataset = dataset[:dropRows]
                lastDate = datetime.datetime.strptime(dataset['indexDate'].values[-1], '%Y-%m-%d')
            num_prediction = (endDate - lastDate).days
            del dataset['indexDate']
            results = common.predict(dataset, startDate, num_prediction, int(request.form['look_back']), int(request.form['batch_size']),os.path.splitext(request.form['filename'])[0])
            response = {}
            response['predictions'] = json.loads(results.to_json(orient='records'))
            return response
    else:
        return "Bad Request", 400
        
if __name__ == '__main__':
    app.debug = True
    app.run(debug=True, port=6001)