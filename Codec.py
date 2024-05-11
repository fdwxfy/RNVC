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


np.random.seed(123)


# In[3]:


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

def build_model():
    model = Sequential()
    model.add(GRU(input_dim=1, units=3, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation('selu'))
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=10e-4), metrics=['mean_absolute_error'])
    return model


# In[5]:


def train_model(train_x, train_y, test_x, test_y):
    model = build_model()

    try:
        model.fit(train_x, train_y, batch_size = 48, epochs = 20, validation_split = 0.1)
        scores = model.evaluate(train_x, train_y, verbose = 0)
        predict = model.predict(test_x)
        predict_y = np.reshape(predict, (predict.size,))
    except KeyboardInterrupt:
        print("predict_y:",predict_y)
        print("test_y",test_y)
    model.save('model2_0.h5')
    return predict_y, test_y



# In[7]:


def dec2bin(x):
    re = []
    x, m = divmod(x, 2)
    re.append(m)
    while x != 0 and len(re) < 3:
        x, m = divmod(x, 2)
        re.append(m)
    while len(re) < 3:
        re.append(0)
    re = list(reversed(re))
    return re


# In[8]:


def quantizer(x):
    scaler = max_err / 4096  # 量化间隔
    pos = []
    if x > 0:
        pos.append(1)
    else:
        pos.append(0)
    pos_ = abs(x) / scaler  # 确定段落
    if pos_ <= 32:
        pos.append(0)
        pos.append(0)
        pos.append(0)
        interval = (32) / 8 # 确定段内间隔
        in_code = math.floor((pos_ - 0) / interval)  # 确定段内码
    elif 32 < pos_ and pos_ <= 64:
        pos.append(0)
        pos.append(0)
        pos.append(1)
        interval = (64 - 32) / 8
        in_code = math.floor((pos_ - 32) / interval)
    elif 64 < pos_ and pos_ <= 128:
        pos.append(0)
        pos.append(1)
        pos.append(0)
        interval = (128 - 64) / 8
        in_code = math.floor((pos_ - 64) / interval)
    elif 128 < pos_ and pos_ <= 256:
        pos.append(0)
        pos.append(1)
        pos.append(1)
        interval = (256 - 128) / 8
        in_code = math.floor((pos_ - 128) / interval)
    elif 256 < pos_ and pos_ <= 512:
        pos.append(1)
        pos.append(0)
        pos.append(0)
        interval = (512 - 256) / 8
        in_code = math.floor((pos_ - 256) / interval)
    elif 512 < pos_ and pos_ <= 1024:
        pos.append(1)
        pos.append(0)
        pos.append(1)
        interval = (1024 - 512) / 8
        in_code = math.floor((pos_ - 512) / interval)
    elif 1024 < pos_ and pos_ <= 2048:
        pos.append(1)
        pos.append(1)
        pos.append(0)
        interval = (2048 - 1024) / 8
        in_code = math.floor((pos_ - 1024) / interval)
    else:
        pos.append(1)
        pos.append(1)
        pos.append(1)
        interval = (4096 - 2048) / 8
        in_code = math.floor((pos_ - 2048) / interval)

    bin = dec2bin(in_code)
    for i in range(len(bin)):
        pos.append(bin[i])
#     matrix = counter(pos)
    return pos


# In[9]:


def I_quantizer(x):
    a = x[0] #首位判断±
    b = x[1] * 4 + x[2] * 2 + x[3] * 1 #2-4位判断段落
    coe = [4, 4, 8, 16, 32, 64, 128, 256] #各个段落的间隔32/8
    base = [0, 32, 64, 128, 256, 512, 1024, 2048]
    c = x[4] * 4 + x[5] * 2 + x[6] * 1 #段内码
    reback = c * coe[b] + base[b] #段内码还原
    fvalue = (reback / 4096) * max_err
    if a == 1:
        value = fvalue
    elif a == 0:
        value = 0 - fvalue
    return value


# In[10]:


def bin2dec(arr):
    s = ''.join('%s' % id for id in (arr[1:]))
    t = int(s, 2)
    if arr[0] == 1:
        t = 0 - t
        dec_ = t
    else:
        dec_ = t
    return dec_


# In[27]:


def bin2dec_7(arr):
    s = ''.join('%s' % id for id in (arr[:]))
    t = int(s, 2)
    return t


# In[29]:


def dec2bin_(x):
    re = []
    if x < 0: re.append(1)
    else: re.append(0)
    x = abs(x)
    str_ =  bin(x).replace('0b','')
    len_ = len(str_)
    while len_ < 4:
        re.append(0)
        len_ += 1
    for i in range(len(str_)):
        re.append(int(str_[i]))
    return re




# In[31]:


#路径
def findcsv(path, ret):
    """Finding the *.txt file in specify path"""
    filelist = os.listdir(path)
    filelist.sort(key=lambda x: int(x[:-4]))
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(".csv"):  # Specify to find the csv file.
                ret.append(de_path)
        else:
            findcsv(de_path, ret)


# In[32]:


rootdir = "D:\\csv"
ret = []
findcsv(rootdir, ret)


# In[33]:

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

retMerge = [retMerge0, retMerge1, retMerge2, retMerge3, retMerge4]
retLeft = [retLeft0, retLeft1, retLeft2, retLeft3, retLeft4]


# In[35]:


GROUP=1
PSNR = []
SNR = []
CR = []
Max_Err = []
Length = []
NewLength = []


for max_err in range(2,40, 2):
    for nums in range(56):
        test_data = np.array(pd.read_csv(retLeft[GROUP][nums], sep=',', usecols=[0])) #测试数据
        label_data = copy.deepcopy(test_data)
        test_data_ = []
        test_data = []
        test_data_.append(list(label_data[0]))
        test_data_.append(list(label_data[1]))
        test_data.append(test_data_)
        test_data = np.array(test_data)
        length = len(label_data) #保存数据长度
        Length.append(length)

        all_err = []
        model = load_model("D:\\models\\model1_1.h5")
        pre_dict = []
        count_err = []
        count_err_ = []
        err = [0]
        pre_dict.append(list(test_data[0][0]))
        pre_dict.append(list(test_data[0][1]))

        allErr = []


        for i in range(len(label_data) - 2):
            predict = list(model.predict(test_data))
            for j in range(1):
                err[j] = label_data[i + 2][j] - predict[0][j]

            q_err_0 = quantizer(err[0])

            allErr.append(q_err_0)

            err[0] = I_quantizer(q_err_0)

            predict[0][0] += err[0]

            test_data[0][0] = test_data[0][1]
            test_data[0][1] = predict[0]
            pre_dict.append(list(predict[0]))

        p_x = []
        l_x = []

        for i in range(len(pre_dict)):
            p_x.append(pre_dict[i][0])

        DeVal = []
        for i in range(len(allErr)):
            DeVal.append(bin2dec_7(allErr[i]))
        dictHuffman = np.array(pd.read_csv("D:\\NNewDict.csv", sep=',', usecols=[0, 1]))
        newLength = 0
        for i in range(len(DeVal)):
            newLength = newLength + dictHuffman[DeVal[i]][1]
        NewLength.append(newLength)
        ratio = length * 16 / newLength
        CR.append(ratio)
        Max_Err.append(max_err)
    print("max_err", max_err)
    print("CR:", CR)







