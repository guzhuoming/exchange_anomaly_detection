import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans, DBSCAN, Birch, OPTICS
from matplotlib import pyplot
import sklearn

import matplotlib.dates as mdates
# plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
font1 = {'family': 'Microsoft YaHei',
         'weight': 'normal',
         'size': 13}

plt.style.use(['science','no-latex'])

exchanges = ["binance", "coinbase", "huobi", "kraken", "kucoin"]

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

def cluster():
    # method = "K-Means"
    # method = "DBSCAN"
    # method = "Birch"
    method = "OPTICS"
    for i in range(len(exchanges)):
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

        df['transaction_amount_usd']
        tempx = df['transaction_amount_usd'].values
        x = []
        for j in range(train_size, time_len):
            x.append([j,tempx[j]])
        x = np.array(x)

        # 定义模型
        if method=="K-Means":
            model = KMeans(n_clusters=3)
        elif method=="DBSCAN":
            model = DBSCAN(eps=1000, min_samples=5)
        elif method=="Birch":
            model = Birch(threshold=0.5, n_clusters=3)
        elif method=="OPTICS":
            model = OPTICS()
        # 模型拟合
        model.fit(x)
        # 为每个示例分配一个集群
        if method=="DBSCAN" or method=="OPTICS":
            yhat = model.fit_predict(x)
        else:
            yhat = model.predict(x)
        # 检索唯一群集
        clusters = unique(yhat)
        pyplot.figure(figsize=(6,4),dpi=300)
        # 为每个群集的样本创建散点图
        for  cluster in clusters:
            row_ix = where(yhat==cluster)
            print(row_ix)
            pyplot.scatter(x[row_ix, 0], x[row_ix, 1])
            pyplot.xlabel("时间", font1)
            pyplot.ylabel("交易量（美元）", font1)
        pyplot.savefig("./exchange/figure/"+method+"/"+exchange+".png")
        pyplot.show()
        # elif method=="DBSCAN":
        #
        # elif method=="GMM":
        #
        # else:
        #     print("Default")

if __name__=='__main__':
    cluster()