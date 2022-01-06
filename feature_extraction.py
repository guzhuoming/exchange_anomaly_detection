"""
features extraction:
transaction num, transaction sum, transaction mean, transaction variance
"""
import pandas as pd
import csv
import numpy as np
import os
from collections import defaultdict
import networkx as nx

file = open('./address.csv')
df = pd.read_csv(file)
address = df['address']

def feature_extraction(ts = 80, minTime=1590940800, n=1):
    """

    :return:
    """
    # create feature files
    for i in range(len(address)):
        if not os.path.exists('./data/feature_4_{}'.format(n)):
            os.makedirs('./data/feature_4_{}'.format(n))
        file2 = open('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]), 'w',
                    newline='')
        csvwriter = csv.writer(file2)
        csvwriter.writerow(['tran_num', 'tran_sum', 'tran_mean', 'tran_var'])

        for j in range(ts):
            csvwriter.writerow([0. for i in range(4)])
        file2.close()

    for i in range(len(address)):
        print('i={}'.format(i))
        node = address[i]
        data = open('./data/source_data/{}.csv'.format(node))
        df_data = pd.read_csv(data)

        # save transaction values
        tran = [[] for i in range(ts)]

        for j in range(len(df_data)):
            ft = open('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]))
            df_ft = pd.read_csv(ft)
            ft.close()

            t = (df_data['TimeStamp'][j] - minTime) // (86400 * n)
            if n==1:
                t-=21
            elif n==3:
                t-=7
            if t>=0 and t<ts :
                df_ft['tran_num'][t] = df_ft['tran_num'][t] + 1
                df_ft['tran_sum'][t] = df_ft['tran_sum'][t] + df_data['Value'][j]

                tran[t].append(df_data['Value'][j])

            df_ft.to_csv('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]),
                         index=False)
        for t in range(ts):
            if len(tran[t])>0:
                df_ft['tran_mean'][t] = np.mean(tran[t])
                df_ft['tran_var'][t] = np.var(tran[t])

            else:
                df_ft['tran_mean'][t] = 0
                df_ft['tran_var'][t] = 0

        df_ft.to_csv('./data/feature_4_{}/{}_ft.csv'.format(n, address[i]),
                     index=False)

def construct_graph(minTime=1590940800, n=1):
    """
    construct graph for each timestep(one day)
    :return:
    """
    graph_list = [defaultdict(list) for i in range(80)]

    address2label = {}
    for i in range(len(address)):
        address2label[address[i]] = i

    for i in range(len(address)):
        print('i = {}'.format(i))
        node = address[i]
        data = open('./data/source_data/{}.csv'.format(node))
        df_data = pd.read_csv(data)

        for j in range(len(df_data)):
            t = (df_data['TimeStamp'][j] - minTime) // (86400 * n)
            t -= 21
            if t>=80:
                break

            x = df_data['From'][j]
            y = df_data['To'][j]

            if x==node:
                if y in address.to_list():
                    temp = address2label[y]
                    if temp not in graph_list[t][i]:
                        graph_list[t][i].append(temp)
            elif y==node:
                if x in address.to_list():
                    temp = address2label[x]
                    if temp not in graph_list[t][i]:
                        graph_list[t][i].append(temp)
            else:
                print('error')
    np.save('./data/graph_list.npy', graph_list)
    print(graph_list)

def save_feature2npy(n_timesteps=80):
    for i in range(n_timesteps):
        temp = []
        for j in range(len(address)):
            feature_file = open('./data/feature_4_1/{}_ft.csv'.format(address[j]))
            df_feature = pd.read_csv(feature_file)
            temp.append(np.array(df_feature.iloc[i]))

        temp = np.array(temp)
        if os.path.exists('./data/feature_npy'):
            np.save('./data/feature_npy/{}.npy'.format(i), temp)
        else:
            os.makedirs('./data/feature_npy')
            np.save('./data/feature_npy/{}.npy'.format(i), temp)

if __name__=='__main__':
    # feature_extraction(ts = 80, minTime=1590940800, n=1)
    # construct_graph()

    # graph_list = np.load('./data/graph_list.npy', allow_pickle=True)
    # for i in range(len(graph_list)):
    #     print(graph_list[i])

    save_feature2npy()