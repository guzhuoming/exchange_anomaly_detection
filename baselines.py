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
from pmdarima.arima import auto_arima
from pmdarima.arima import ADFTest
import matplotlib.dates as mdates

plt.style.use(['science','no-latex'])

# plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
font1 = {'family': 'Microsoft YaHei',
         'weight': 'normal',
         'size': 13}

exchanges = ["binance", "coinbase", "huobi", "kraken", "kucoin"]
error_list = []
method = "HA"

def data_split(data, train_rate, seq_len, pre_len=1):
    time_len, n_feature = data.shape
    train_size = int(time_len * train_rate)
    print("time_len = {}".format(time_len))
    print("train_size = {}".format(train_size))
    print("test_size = {}".format(time_len-train_size))

    # 交易所    开始时间     time_len   train_size    test_size   测试集开始时间    测试集结束时间
    # Binance  1499212800   1632        1305           327       2021-01-30      2021-12-23
    # Coinbase 1619481600   241         192            49        2021-11-05      2021-12-24
    # Huobi    1495497600   1576        1260           316       2020-11-03      2021-09-15
    # Kraken   1438905600   2231        1784           447       2020-06-25      2021-09-15
    # Kucoin   1505606400   1559        1247           312       2021-02-15      2021-12-24

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

def baseline():
    li = []
    # li.append([1400, 1415, 1416, 1420, 1421, 1543])
    # li.append([222])
    # li.append([1413, 1450, 1501])
    # li.append([2097, 2111, 2113])
    # li.append([1463, 1503, 1522, 1542])

    li.append([95, 110, 111, 115, 116, 238])
    li.append([30])
    li.append([153, 190, 241])
    li.append([313, 327, 329])
    li.append([216, 256, 275, 295])

    # 1305  lstm:  ha:95,115,238
    # 192   lstm:  ha:30
    # 1260  lstm:  ha:153,190,241
    # 1784  lstm:  ha:313
    # 1247  lstm:  ha:256,275,295

    beginDates = ['2021-01-30', '2021-11-05', '2020-11-03', '2020-06-25', '2021-02-15']
    endDates = ['2021-12-23', '2021-12-24', '2021-09-15', '2021-09-15', '2021-12-24']

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
        # 原始的数据
        trainX, trainY, testX, testY = data_split(data, train_rate=train_rate, seq_len=seq_len)
        scaled_data = data.copy()

        scaler = []
        for j in range(n_features):
            temp_scaler = MinMaxScaler(feature_range=(0, 1))
            temp = temp_scaler.fit_transform(np.array(scaled_data[:, j]).reshape(-1, 1))
            scaler.append(temp_scaler)
            temp = temp.reshape(-1)
            scaled_data[:, j] = temp
        # 归一化的数据
        trainX1, trainY1, testX1, testY1 = data_split(scaled_data, train_rate=train_rate, seq_len=seq_len)

        if method == "HA":
            prediction_val = []
            for j in range(len(testX1)):
                a = np.array(testX1[j])
                # print("a.shape")
                # print(a.shape)
                prediction_val.append(np.mean(a[:,0])) #只对第0列特征（交易量）求平均
            print(len(prediction_val))
            print(len(testY1))
            rmse, mae, mape, r2, var, _ = evaluation(testY1, prediction_val)
            print("exchange = {}".format(exchange))
            print("rmse = {}\nmae = {}\nmape = {}\nr2 = {}\nvar = {}\n".format(rmse, mae, mape, r2, var))
            error_list.append("rmse = {}\nmae = {}\nmape = {}\n".format(rmse, mae, mape))

            prediction_val = np.array(scaled_data[0:train_size, 0].tolist() + prediction_val)
            prediction_val = prediction_val.reshape(-1, 1)
            prediction_val = scaler[0].inverse_transform(prediction_val)
            prediction_val = prediction_val[train_size:, 0]
            testY = testY[:, 0]
            upper_bound = [it + 3 * np.std(testY) for it in prediction_val]
            lower_bound = [it - 3 * np.std(testY) for it in prediction_val]

            # 打印miss
            # for j in range(len(upper_bound)):
            #     if testY[j] > upper_bound[j]:
            #         abnormal_x.append(j)
            #         abnormal_y.append(testY[j])


            fig, ax = plt.subplots(figsize=(10, 4))
            beginDate = beginDates[i]
            endDate = endDates[i]


            x = np.arange(mdates.datestr2num(beginDate), mdates.datestr2num(endDate))
            x_range = [np.datetime64(int(c), 'D') for c in x]
            # plt.figure(figsize=(10, 4))
            # plt.grid(linestyle="--")

            abnormal_x = []
            abnormal_y = []
            for j in range(len(upper_bound)):
                if testY[j] > upper_bound[j]:
                    abnormal_x.append(x_range[j])
                    abnormal_y.append(testY[j])

            miss_x = []
            miss_y = []
            for j in range(len(li[i])):
                if x_range[li[i][j]] not in abnormal_x:
                    miss_x.append(x_range[li[i][j]])
                    miss_y.append(testY[li[i][j]])


            """设置坐标轴的格式"""
            # 设置主刻度, 每6个月一个刻度
            fmt_half_year = mdates.MonthLocator(interval=1)
            ax.xaxis.set_major_locator(fmt_half_year)

            # 设置次刻度，每个月一个刻度
            fmt_month = mdates.MonthLocator()  # 默认即可
            ax.xaxis.set_minor_locator(fmt_month)

            # 设置 x 坐标轴的刻度格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

            # 设置横坐标轴的范围
            datemin = np.datetime64(x_range[0], 'M')
            # datemax = np.datetime64(x_range[-1], 'Y') + np.timedelta64(1, 'Y')
            datemax = np.datetime64(x_range[-1], 'M') + np.timedelta64(1, 'M')
            ax.set_xlim(datemin, datemax)

            # 设置刻度的显示格式
            ax.format_xdata = mdates.DateFormatter('%Y-%m')
            ax.format_ydata = lambda x: f'$x:.2f$'
            ax.grid(True)
            """自动调整刻度字符串"""
            # 自动调整 x 轴的刻度字符串（旋转）使得每个字符串有足够的空间而不重叠
            fig.autofmt_xdate()

            ax.plot(x_range, testY)
            ax.plot(x_range, np.array(prediction_val))
            ax.plot(x_range, upper_bound, "--")
            ax.plot(x_range, lower_bound, "--")
            ax.scatter(abnormal_x, abnormal_y, c="r", marker="o")
            ax.scatter(miss_x, miss_y, c="dimgrey", marker="o")
            plt.legend(("Real", method, "Upper bound", "Lower bound", "Anomaly detected", "Anomaly missed"), loc=2)
            plt.title(exchange.title())
            plt.xlabel("时间",font1)
            # plt.ylabel("Transaction Amount(USD)")
            plt.ylabel("交易量（美元）",font1)
            plt.savefig('./exchange/figure/ha_prediction' + '/' + exchange + '_ha.png')
            plt.show()
        if method == 'ARIMA':
            if False:
                #自动选择参数的arima，用来为每一个交易所选择最优参数，但是预测效果不好
                #原因可能是没有用到测试时间段的信息，所以选择最优参数用后面的arima代码来预测
                temp_data = df['transaction_amount_usd']

                adf_test = ADFTest(alpha=0.05)
                print(adf_test.should_diff(temp_data))
                # 仅使用训练时间段
                # binance
                # (0.01, False)
                # coinbase
                # (0.1639271547588751, True)
                # huobi
                # (0.01, False)
                # kraken
                # (0.038886500597949056, False)
                # kucoin
                # (0.21528544528481058, True)
                # True：是平稳序列，不需要差分d=0， False：不是平稳序列，需要差分，d!=0

                train = temp_data[:train_size]
                test = temp_data[train_size:]
                prediction = []
                if i==0 or i==2 or i==3:
                    arima_model = auto_arima(temp_data, start_p=0, d=1, start_q=0,
                                             max_p=5, max_d=5, max_q=5, start_P=0,
                                             D=0, start_Q=0, max_P=5, max_D=5,
                                             max_Q=5, m=12, seasonal=False)
                    print(arima_model.summary())
                    prediction = arima_model.predict(n_periods=time_len-train_size)

                else:
                    arima_model = auto_arima(temp_data, start_p=0, d=0, start_q=0,
                                             max_p=5, max_d=5, max_q=5, start_P=0,
                                             D=0, start_Q=0, max_P=5, max_D=5,
                                             max_Q=5, m=12, seasonal=False)
                    print(arima_model.summary())
                    prediction = arima_model.predict(n_periods=time_len-train_size)

                plt.figure()
                # plt.plot(test)
                # plt.plot(prediction)
                # plt.plot(range(len(temp_data)),temp_data)
                # plt.plot(train)
                # plt.plot(test)
                plt.show()
                # refer to https://towardsdatascience.com/time-series-forecasting-using-auto-arima-in-python-bb83e49210cd
            if True:#自己选择参数的arima
                temp_data = data[:,0]
                temp_temp_scaler = MinMaxScaler(feature_range=(0, 1))
                temp_data = temp_temp_scaler.fit_transform(temp_data.reshape(-1,1))
                history = (temp_data[0:train_size].reshape(1,-1))[0].tolist()
                print(history)
                test = testY1.reshape(1,-1)[0]
                print(test)
                pred = []

                p, d, q = 1,1,1

                # if i==0:
                #     p, d, q = 2, 1, 3
                # elif i==1:
                #     p, d, q = 1, 0, 2
                # elif i==2:
                #     p, d, q = 0, 1, 3
                # elif i==3:
                #     p, d, q = 2, 1, 3
                # elif i==4:
                #     p, d, q = 2, 0, 2

                # if i==0:
                #     p, d, q = 5, 1, 1
                # elif i==1:
                #     p, d, q = 1, 0, 2
                # elif i==2:
                #     p, d, q = 1, 1, 4
                # elif i==3:
                #     p, d, q = 3, 1, 3
                # elif i==4:
                #     p, d, q = 5, 0, 1

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
                    # history.append(yhat[0])
                rmse, mae, mape, r2, var, _ = evaluation(test, pred)
                print("rmse = {}\nmae = {}\nmape = {}\nr2 = {}\nvar = {}\n".format(rmse, mae, mape, r2, var))
                error_list.append("rmse = {}\nmae = {}\nmape = {}\n".format(rmse, mae, mape))

                fig, ax = plt.subplots(figsize=(10, 4))
                beginDate = beginDates[i]
                endDate = endDates[i]
                x = np.arange(mdates.datestr2num(beginDate), mdates.datestr2num(endDate))
                x_range = [np.datetime64(int(c), 'D') for c in x]

                pred = temp_temp_scaler.inverse_transform(np.array(pred).reshape(-1, 1))
                test = temp_temp_scaler.inverse_transform(np.array(test).reshape(-1, 1))

                upper_bound = [it + 3 * np.std(test) for it in pred]
                lower_bound = [it - 3 * np.std(test) for it in pred]

                abnormal_x = []
                abnormal_y = []
                for j in range(len(upper_bound)):
                    if testY[j] > upper_bound[j]:
                        abnormal_x.append(x_range[j])
                        abnormal_y.append(testY[j])

                miss_x = []
                miss_y = []
                for j in range(len(li[i])):
                    if x_range[li[i][j]] not in abnormal_x:
                        miss_x.append(x_range[li[i][j]])
                        miss_y.append(testY[li[i][j]])

                """设置坐标轴的格式"""
                # 设置主刻度, 每6个月一个刻度
                fmt_half_year = mdates.MonthLocator(interval=1)
                ax.xaxis.set_major_locator(fmt_half_year)

                # 设置次刻度，每个月一个刻度
                fmt_month = mdates.MonthLocator()  # 默认即可
                ax.xaxis.set_minor_locator(fmt_month)

                # 设置 x 坐标轴的刻度格式
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

                # 设置横坐标轴的范围
                datemin = np.datetime64(x_range[0], 'M')
                # datemax = np.datetime64(x_range[-1], 'Y') + np.timedelta64(1, 'Y')
                datemax = np.datetime64(x_range[-1], 'M') + np.timedelta64(1, 'M')
                ax.set_xlim(datemin, datemax)

                # 设置刻度的显示格式
                ax.format_xdata = mdates.DateFormatter('%Y-%m')
                ax.format_ydata = lambda x: f'$x:.2f$'
                ax.grid(True)
                """自动调整刻度字符串"""
                # 自动调整 x 轴的刻度字符串（旋转）使得每个字符串有足够的空间而不重叠
                fig.autofmt_xdate()

                ax.plot(x_range, testY)
                ax.plot(x_range, np.array(pred))
                ax.plot(x_range, upper_bound, "--")
                ax.plot(x_range, lower_bound, "--")
                ax.scatter(abnormal_x, abnormal_y, c="r", marker="o")
                ax.scatter(miss_x, miss_y, c="dimgrey", marker="o")
                plt.legend(("Real", method, "Upper bound", "Lower bound", "Anomaly detected", "Anomaly missed"), loc=2)
                plt.title(exchange.title())
                plt.xlabel("时间", font1)
                # plt.ylabel("Transaction Amount(USD)")
                plt.ylabel("交易量（美元）", font1)
                plt.savefig('./exchange/figure/arima_prediction' + '/' + exchange + '_arima.png')
                plt.show()
        if method == 'SVR':
            # print("trainX.shape")

            temp_data = data[:, 0]
            temp_temp_scaler = MinMaxScaler(feature_range=(0, 1))
            temp_data = temp_temp_scaler.fit_transform(temp_data.reshape(-1, 1))

            trainX = trainX[:, :, 0]
            trainX1 = trainX1[:, :, 0]
            testX = testX[:, :, 0]
            testX1 = testX1[:, :, 0]

            print(trainX.shape)
            # print(trainY.shape)
            print(testX.shape)
            # print(testY.shape)
            # binance
            # (1294, 10, 15)
            # (1294, 1)
            # (327, 10, 15)
            # (327, 1)

            # print("trainX.type")
            # print(type(trainX))
            # print(type(trainY))
            # print(type(testX))
            # print(type(testY))

            # print(trainX)
            # for t in range(len(test)):
            #     model = ARIMA(history, order=(p, d, q))
            #     model_fit = model.fit()
            #     output = model_fit.forecast()
            #     yhat = output[0]
            #     if yhat[0] < 0:
            #         print("ding")
            #         yhat[0] = 0
            #     pred.append(yhat[0])
            #     history.append(test[t])
            pred = []
            for t in range(len(testX)):
                # print("t = {}".format(t))

                svr_model = SVR(kernel='rbf')
                # svr_model.fit(trainX1, trainY1)
                # pred = svr_model.predict(testX1)
                # svr_model.fit(np.concatenate([trainX1, testX1]), np.concatenate([trainY1, testY1]))
                svr_model.fit(trainX1, trainY1)
                output = svr_model.predict(testX1[t:])

                yhat = output[0]
                if yhat < 0:
                    print("ding")
                    yhat = 0
                pred.append(yhat)

                trainX1 = np.concatenate([trainX1, testX1[t:t+1]])
                trainY1 = np.concatenate([trainY1, testY1[t:t+1]])

            pred = np.array(pred)
            rmse, mae, mape, r2, var, _ = evaluation(testY1, pred)
            print("rmse = {}\nmae = {}\nmape = {}\nr2 = {}\nvar = {}\n".format(rmse, mae, mape, r2, var))

            pred = temp_temp_scaler.inverse_transform(pred.reshape(-1, 1))

            fig, ax = plt.subplots(figsize=(10, 4))
            beginDate = beginDates[i]
            endDate = endDates[i]
            # pred = temp_temp_scaler.inverse_transform(np.array(pred).reshape(-1, 1))
            # test = temp_temp_scaler.inverse_transform(np.array(testY1).reshape(-1, 1))
            x = np.arange(mdates.datestr2num(beginDate), mdates.datestr2num(endDate))
            x_range = [np.datetime64(int(c), 'D') for c in x]

            upper_bound = [it + 3 * np.std(testY) for it in pred]
            lower_bound = [it - 3 * np.std(testY) for it in pred]

            abnormal_x = []
            abnormal_y = []
            for j in range(len(upper_bound)):
                if testY[j] > upper_bound[j]:
                    abnormal_x.append(x_range[j])
                    abnormal_y.append(testY[j])

            miss_x = []
            miss_y = []
            for j in range(len(li[i])):
                if x_range[li[i][j]] not in abnormal_x:
                    miss_x.append(x_range[li[i][j]])
                    miss_y.append(testY[li[i][j]])

            """设置坐标轴的格式"""
            # 设置主刻度, 每6个月一个刻度
            fmt_half_year = mdates.MonthLocator(interval=1)
            ax.xaxis.set_major_locator(fmt_half_year)

            # 设置次刻度，每个月一个刻度
            fmt_month = mdates.MonthLocator()  # 默认即可
            ax.xaxis.set_minor_locator(fmt_month)

            # 设置 x 坐标轴的刻度格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

            # 设置横坐标轴的范围
            datemin = np.datetime64(x_range[0], 'M')
            # datemax = np.datetime64(x_range[-1], 'Y') + np.timedelta64(1, 'Y')
            datemax = np.datetime64(x_range[-1], 'M') + np.timedelta64(1, 'M')
            ax.set_xlim(datemin, datemax)

            # 设置刻度的显示格式
            ax.format_xdata = mdates.DateFormatter('%Y-%m')
            ax.format_ydata = lambda x: f'$x:.2f$'
            ax.grid(True)
            """自动调整刻度字符串"""
            # 自动调整 x 轴的刻度字符串（旋转）使得每个字符串有足够的空间而不重叠
            fig.autofmt_xdate()

            ax.plot(x_range, testY)
            ax.plot(x_range, np.array(pred))
            ax.plot(x_range, upper_bound, "--")
            ax.plot(x_range, lower_bound, "--")
            ax.scatter(abnormal_x, abnormal_y, c="r", marker="o")
            ax.scatter(miss_x, miss_y, c="dimgrey", marker="o")
            plt.legend(("Real", method, "Upper bound", "Lower bound", "Anomaly detected", "Anomaly missed"), loc=2)
            plt.title(exchange.title())
            plt.xlabel("时间", font1)
            # plt.ylabel("Transaction Amount(USD)")
            plt.ylabel("交易量（美元）", font1)
            plt.savefig('./exchange/figure/svr_prediction' + '/' + exchange + '_svr.png')
            plt.show()

# for i in range(len(exchanges)):
#     print(exchanges[i])
#     print(error_list[i])

if __name__=='__main__':
    baseline()