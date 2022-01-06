from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.layers import Input, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, Layer
from keras.layers import Concatenate, Reshape
import keras

from graph import GraphConvolution
from utils import *
from graph_attention_layer import GraphAttention
import scipy
import pandas as pd

import time

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

from temporal import attention

import tensorflow
tensorflow.set_random_seed(2)

data = open('./address.csv')
df = pd.read_csv(data)
address = df['address']

"""
    :parameter
"""
n_timesteps = 80
n_nodes = len(address)
n_train = 60
n_test = n_timesteps-n_train
n_window = 10
"""
    construct graph
"""

Graph_list = np.load('./data/graph_list.npy', allow_pickle=True)
A_list = [np.zeros((n_nodes, n_nodes)) for i in range(n_timesteps)] # A: adjacency matrix
for i in range(n_timesteps):
    for j in range(n_nodes):
        if Graph_list[i][j]:
            for k in range(len(Graph_list[i][j])):
                A_list[i][j][Graph_list[i][j][k]] = 1
        else:
            continue
    A_list[i] = scipy.sparse.csr_matrix(A_list[i])
    # A_list[i] = A_list[i] + np.eye(A_list[i].shape[0])
# print(A_list)

"""
    load feature npy files
"""

data = []
for i in range(n_timesteps):
    temp = np.load('./data/feature_npy/{}.npy'.format(i))
    temp = temp.tolist()
    temp = np.mat(temp)
    data.append(temp)

N = data[0].shape[0]                # Number of nodes in the graph
F = data[0].shape[1]                # Original feature dimension
F_ = F                   # Output size of first GraphAttention layer
"""
    extract transum as y_train 
"""
dataset_total = []
for i in range(len(address)):
    file = open('./data/feature_4_1/{}_ft.csv'.format(address[i]))
    df = pd.read_csv(file)
    new_data = pd.DataFrame(df, columns=['tran_sum'])
    dataset = new_data.values
    dataset = dataset.reshape(1, -1)
    dataset_total.append(dataset)

dataset_total = np.array(dataset_total)
dataset_total = np.reshape(dataset_total, (N, n_timesteps))
# print(dataset_total)
# print(dataset_total.shape) #(15, 80)
y_train = []
for i in range(n_train):
    temp = [a[1+i] for a in dataset_total]
    y_train.append(temp)
y_train = np.array(y_train)
y_train = scaler.fit_transform(y_train)
# print(y_train.shape) #(50, 15)

"""
    real transum
"""
transum = []
for i in range(n_test): # 20test samples
    temp = [a[n_train+i] for a in dataset_total]
    transum.append(temp)
transum = np.array(transum)
print(transum)
# scaler.fit_transform(transum)

def gcn_lstm(with_att=False, n_epochs=20):
    FILTER = 'localpool'  # 'chebyshev'
    MAX_DEGREE = 2  # maximum polynomial degree
    SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
    if FILTER == 'localpool':
        A_list_ = []
        graph_all = []
        for i in range(n_timesteps):
            A_ = preprocess_adj(A_list[i], SYM_NORM)
            A_list_.append(A_)
            graph = [data[i], A_]
            graph_all.append(graph)
        support = 1
        G = [Input(shape=(None, None), batch_shape=(None, None), sparse=True)]
    else:
        raise Exception('Invalid filter type.')

    data_train = []
    data_test = []
    for i in range(n_train):
        data_train.append(graph_all[i])

    for i in range(n_train-1, n_timesteps-1):
        data_test.append(graph_all[i])

    """
        construct the model
    """
    X_in = Input(shape=(F,), batch_shape=(N, F))
    # H = Dropout(0.5)(X_in)
    H = GraphConvolution(F_, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in] + G)
    H = Reshape((-1, F))(H)
    return_sequence = False
    if with_att == True:
        return_sequence = True
    gcn_lstm = LSTM(units=50, input_shape=[N, F], return_sequences=return_sequence)(H)
    if with_att==True:
        gcn_lstm = attention()(gcn_lstm)
    gcn_lstm = Dense(1)(gcn_lstm)

    # Compile model
    model = Model(inputs=[X_in] + G, outputs=gcn_lstm)
    # model = Model(inputs=[[X_in]+G, A_in], outputs=gcn_lstm)
    model.compile(loss='mse', optimizer='adam')

    for i in range(n_train):
        model.fit(data_train[i], y_train[i], batch_size=N, epochs=n_epochs)

    value = []
    for i in range(n_test):
        value.append(model.predict(data_test[i], batch_size=N))

    value = np.array(value).reshape(n_test, N)
    print(value)
    value = scaler.inverse_transform(value)
    value[value < 0] = 0
    print(value)

    rmse = 0
    rmse = np.sqrt(np.mean(np.power((value - transum), 2)))
    print(rmse)
    return rmse

if __name__=='__main__':
    gcn_lstm(with_att=False)