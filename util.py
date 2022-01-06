import pandas as pd
import numpy as np
import math
from math import sqrt
import os
import csv
from matplotlib import pyplot as plt

file = open('./address.csv')
df = pd.read_csv(file)
address = df['address']

def plt_ground_truth():
    for i in range(len(address)):
        node = address[i]
        data = open('./data/feature_4_1/{}_ft.csv'.format(node))
        df_data = pd.read_csv(data)

        ground_truth = df_data['tran_sum'].values.tolist()

        x = range(len(ground_truth))

        plt.plot(x, ground_truth)
        plt.title(address[i])
        plt.savefig('./figure/ground truth/{}.png'.format(address[i]), dpi=600)
        plt.show()

if __name__=='__main__':
    plt_ground_truth()