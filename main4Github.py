from keras.layers import Input,Dense,Lambda,LSTM,MaxPooling1D,AveragePooling1D,AveragePooling2D,Conv2D,Reshape,add,concatenate,BatchNormalization,Dropout,Activation
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import math
import scipy.io as sio
import numpy as np
import keras
import time
time_batch=5#here you should input your window size
dim=60#here you should input your latent dimension
data=sio.loadmat('ywbdata_92av3c5.mat')
indexi=data['indexi'].astype('float32')
indexj=data['indexj'].astype('float32')
X_LSTM=data['X_LSTM'].astype('float32')
X_LSTM=X_LSTM*2-1
shuffle=1
if shuffle==1:
    index=np.arange(10366)
    np.random.seed(1)
    np.random.shuffle(index)
    X_LSTM=X_LSTM[index]
    indexi=indexi[index]
    indexj=indexj[index]
real_batch=time_batch*time_batch
batchsize=512#here you should input your batchsize
original_dim=X_LSTM.shape[3]
nb_epoch=5
epsilon_std=1
###############################################################################
#The code of the network would be provided latter.
###############################################################################