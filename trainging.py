#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Activation,Dropout,GRU,SimpleRNN
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import math
import copy
import csv
from math import sqrt
import os


# In[2]:


#随机数种子
np.random.seed(123)


# In[3]:


#将数据进行处理
def load_data(arr, sequence_length=2, split=0.8):
    data_all = list(np.array(arr).astype(float))
    data = []
    for i in range(len(data_all) - sequence_length - 1): #最大化数据集
        data.append(data_all[i:i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[:split_boundary]
    val_x = x[split_boundary:]
    train_y = y[:split_boundary]
    val_y = y[split_boundary:]
    print("train_x",train_x)
    return train_x, train_y, val_x, val_y


# In[4]:


#搭建模型
def build_model():
    model = Sequential()
    model.add(GRU(input_dim=1, units=3, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('selu'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=10e-4), metrics=['mean_absolute_error'])
    return model


# In[5]:


#训练模型
def train_model(train_x, train_y, test_x, test_y):
    model = build_model()

    try:
        model.fit(train_x, train_y, batch_size = 48, epochs = 30, validation_split = 0.1)
        scores = model.evaluate(train_x, train_y, verbose = 0)
        predict = model.predict(test_x)
        predict_y = np.reshape(predict, (predict.size,))
    except KeyboardInterrupt:
        print("predict_y:",predict_y)
        print("test_y",test_y)
    model.save('lstm_model2_0.h5')
    return predict_y, test_y

# In[31]:


#路径
def findcsv(path, ret):
    """Finding the *.txt file in specify path"""
    filelist = os.listdir(path)
    filelist.sort(key=lambda x: int(x[:-4]))  # 数据集按照序号排列
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(".csv"):  # Specify to find the csv file.
                ret.append(de_path)
        else:
            findcsv(de_path, ret)


# In[32]:


# 数据集路径
rootdir = "F:\\csv_n"
ret = []
findcsv(rootdir, ret) #数据路径   path的路径都存放在ret，将所有包含的路径文件划分为5个ret


# In[33]:

#五折交叉验证
#将数据集分为五份
ret0 = []
ret1 = []
ret2 = []
ret3 = []
ret4 = []
for i in range(0, 280, 5):
    ret0.append(ret[i])
    ret1.append(ret[i + 1])
    ret2.append(ret[i + 2])
    ret3.append(ret[i + 3])
    ret4.append(ret[i + 4])

retMerge0 = ret0 + ret1 + ret2 + ret3
retLeft0 = ret4
retMerge1 = ret0 + ret1 + ret2 + ret4
retLeft1 = ret3
retMerge2 = ret0 + ret1 + ret3 + ret4
retLeft2 = ret2
retMerge3 = ret0 + ret2 + ret3 + ret4
retLeft3 = ret1
retMerge4 = ret1 + ret2 + ret3 + ret4
retLeft4 = ret0

retMerge = [retMerge0, retMerge1, retMerge2, retMerge3, retMerge4]  #训练集
retLeft = [retLeft0, retLeft1, retLeft2, retLeft3, retLeft4]        #验证集


# In[35]:



#for GROUP in range(5):


GROUP=1   #GROUP 从0－4设置

   #训练数据
train_data = [0]
for i in range(224):
    train_data_new = np.array(pd.read_csv(retMerge[GROUP][i], sep=',', usecols=[0]))
    train_data = np.vstack((train_data, train_data_new))
train_data = train_data[1:] #删除第一行,得到训练数据
        #将小于0.001的数值置0.001
for i in range(train_data.shape[0]):
    for j in range(train_data.shape[1]):
        if abs(train_data[i][j]) < 0.001:
            train_data[i][j] = 0.001


train_x, train_y, val_x, val_y = load_data(train_data, sequence_length=2, split=0.8)
        #模型训练
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1], 1))
predict_y, val_y = train_model(train_x, train_y, val_x, val_y)
predict_y = np.reshape(predict_y, (-1, 1))








