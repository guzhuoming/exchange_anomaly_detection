import pandas as pd
import numpy as np
import math
from math import sqrt
import os
import csv
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy.linalg as la
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

exchanges = ["binance", "coinbase", "huobi", "kraken", "kucoin"]
error_list = []
method = "HA"

def data_split(data, train_rate, seq_len, pre_len=1):
    time_len, n_feature = data.shape
    train_size = int(time_len * train_rate)
    train_data = data[0:train_size]
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(train_size-seq_len-pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len, 0])
    for i in range(train_size-seq_len-pre_len, time_len-pre_len-seq_len):
        b = data[i: i+seq_len+pre_len,:]
        testX.append(b[0:seq_len])
        testY.append(b[seq_len:seq_len+pre_len, 0])
    trainX1 = np.array(trainX)
    trainY1 = np.array(trainY)
    testX1 = np.array(testX)
    testY1 = np.array(testY)
    return trainX1, trainY1, testX1, testY1

def evaluation(real, pre):
    rmse = mean_squared_error(real, pre, squared=False)
    mae = mean_absolute_error(real, pre)
    mape = mean_absolute_percentage_error(real, pre)
    r2 = r2_score(real, pre)
    var = 1-(np.var(real - pre))/np.var(real)
    F_norm = la.norm(real-pre)/la.norm(real)

    return rmse, mae, mape, r2, var, 1-F_norm

for i in range(len(exchanges)):
    exchange = exchanges[i]
    print("i = {}".format(i))
    print(exchange)
    exchange = exchanges[i]
    file = open('./exchange/feature/' + exchange + '_ft.csv')
    df = pd.read_csv(file)

    data = df.values
    time_len, n_features = data.shape
    train_rate = 0.8
    train_size = int(time_len * train_rate)
    seq_len = 10
    trainX, trainY, testX, testY = data_split(data, train_rate=train_rate, seq_len=seq_len)
    scaled_data = data.copy()

    scaler = []
    for j in range(n_features):
        temp_scaler = MinMaxScaler(feature_range=(0, 1))
        temp = temp_scaler.fit_transform(np.array(scaled_data[:, j]).reshape(-1, 1))
        scaler.append(temp_scaler)
        temp = temp.reshape(-1)
        scaled_data[:, j] = temp
    trainX1, trainY1, testX1, testY1 = data_split(scaled_data, train_rate=train_rate, seq_len=seq_len)

    if method == "HA":
        prediction_val = []
        for j in range(len(testX1)):
            a = np.array(testX1[j])
            prediction_val.append(np.mean(a))
        print(len(prediction_val))
        print(len(testY1))
        rmse, mae, mape, r2, var, _ = evaluation(testY1, prediction_val)
        print("rmse = {}\nmae = {}\nmape = {}\nr2 = {}\nvar = {}\n".format(rmse, mae, mape, r2, var))
        error_list.append("rmse = {}\nmae = {}\nmape = {}\n".format(rmse, mae, mape))

        prediction_val = np.array(scaled_data[0:train_size, 0].tolist() + prediction_val)
        prediction_val = prediction_val.reshape(-1, 1)
        prediction_val = scaler[0].inverse_transform(prediction_val)
        prediction_val = prediction_val[train_size:, 0]
        testY = testY[:, 0]
        plt.figure()
        plt.plot(range(time_len-train_size), testY)
        plt.plot(range(time_len - train_size), np.array(prediction_val))
        plt.legend(("real", method))
        plt.title(exchange)
        plt.show()
    if method == 'ARIMA':
        temp_data = data[:,0]
        temp_temp_scaler = MinMaxScaler(feature_range=(0, 1))
        temp_data = temp_temp_scaler.fit_transform(temp_data.reshape(-1,1))
        history = (temp_data[0:train_size].reshape(1,-1))[0].tolist()
        print(history)
        test = testY1.reshape(1,-1)[0]
        print(test)
        pred = []
        p = 5
        d = 1
        q = 0
        for t in range(len(test)):
            model = ARIMA(history, order=(p,d,q))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            if yhat[0]<0:
                print("ding")
                yhat[0] = 0
            pred.append(yhat[0])
            history.append(test[t])
        rmse, mae, mape, r2, var, _ = evaluation(test, pred)
        print("rmse = {}\nmae = {}\nmape = {}\nr2 = {}\nvar = {}\n".format(rmse, mae, mape, r2, var))
        error_list.append("rmse = {}\nmae = {}\nmape = {}\n".format(rmse, mae, mape))
        plt.figure()
        plt.plot(range(time_len - train_size), test)
        plt.plot(range(time_len - train_size), np.array(pred))
        plt.legend(("real", method))
        plt.title(exchange)
        plt.show()

for i in range(len(exchanges)):
    print(exchanges[i])
    print(error_list[i])