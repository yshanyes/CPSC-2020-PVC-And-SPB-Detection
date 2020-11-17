import warnings
warnings.filterwarnings('ignore')

import matplotlib
import matplotlib.pyplot as plt

import wfdb 
import numpy as np
import math
import sys
import scipy.stats as st
import scipy.io as sio
import scipy
import glob, os
from os.path import basename

import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense,Activation,Dropout,add
from keras.layers import LSTM,Bidirectional #could try TimeDistributed(Dense(...))
from keras.models import Sequential, load_model
from keras import optimizers,regularizers
from keras.layers.normalization import BatchNormalization
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tqdm import tqdm
from keras.utils import np_utils
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt                    #导入pandas包
from keras.utils import to_categorical
from tqdm import tqdm

from scipy import signal
# from biosppy.signals import ecg
import logging


FS_ORI = 400
FS = 200

CONST = FS/FS_ORI
THR = 0.15
DATA_PATH = '../TrainingSet/data/'
REF_PATH = '../TrainingSet/ref/'
ms_150 = 0.12*FS
ms_200 = 0.15*FS
len_seg = FS*10#1024
len_ecg = FS*10#1024


from scipy import ndimage, misc
def med_filt_1d(ecg):
    '''
    first_filtered = ndimage.median_filter(ecg, size=int(0.4*FS))
    second_filtered =  ndimage.median_filter(first_filtered, int(2*FS))
    '''
    first_filtered = ndimage.median_filter(ecg, size=int(7*(FS/FS_ORI)))
    second_filtered =  ndimage.median_filter(first_filtered, int(215*(FS/FS_ORI)))
    ecg_deno = ecg - second_filtered
    return ecg_deno

from scipy.signal import butter, sosfilt, sosfiltfilt, filtfilt
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype="band", output="sos")
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos,data)    
    return y

def butter_bandpass_forward_backward_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos,data) 
    return y

from sklearn import preprocessing as prep

def getSigArr(sig, sigNorm='minmax'):
    if sigNorm == 'scale':
        sig = prep.scale(sig)
    elif sigNorm == 'minmax':
        min_max_scaler = prep.MinMaxScaler()
        sig = min_max_scaler.fit_transform(sig)
    return sig


def firstLabel(label_new, label_ref, len_ecg, label_type): 
    '''
    根据早搏位置，对早搏心拍进行标记
    '''
    try:
        for m in label_ref:
            m_begin = int(m-ms_150 if m-ms_150>=0 else 0)
            m_end = int(m+ms_200 if m+ms_200<len_ecg else len_ecg-1)

            for n in range(m_begin,m_end):
                label_new[n] = label_type
                
    except TypeError:
        print("TypeError")
        m_begin = int(label_ref-ms_150 if label_ref-ms_150>=0 else 0)
        m_end = int(label_ref+ms_200 if label_ref+ms_200<len_ecg else len_ecg-1)

        for n in range(m_begin,m_end):
            label_new[n] = label_type
            
        pass
    return


def resave_data(ecg_data, label_new, dataid, is_select = 0):
    '''
    数据分段，段长度为len_seg，
    如果第一秒和最后一秒数据出现早搏，则将数据前移或者后移一秒
    '''
    mod_res = len(ecg_data)%len_seg 
    if (mod_res > 0):
        num_to_pad = len_seg - mod_res
        ecg_data_pad = np.pad(ecg_data,(0,num_to_pad),'constant', constant_values=(0,0)) 
        label_new_pad = np.pad(label_new,(0,num_to_pad),'constant', constant_values=(0,0)) 
    else:
        ecg_data_pad = ecg_data
        label_new_pad = np.array(label_new)
    
    ecg_data_mat = None
    label_new_mat = None
    
    len_ecg_pad = len(ecg_data_pad)

    ind = 0
    while ind < len_ecg_pad:
        ind_begin = ind
        ind_end = ind_begin+len_seg
        
        if ind_end < len_ecg_pad:
            label_tmp_1 = label_new_pad[ind_begin: ind_end]
            
            if ((np.sum(label_tmp_1)) == 0 and is_select):
                ind = ind_begin+len_seg 
                continue
            else:
                if (is_select):
                    if ((np.sum(label_tmp_1[:FS])) >0) :
                        ind_begin = ind - FS
                    elif ((np.sum(label_tmp_1[-FS:]))>0):
                        ind_begin = ind + FS
                    ind_end = ind_begin+len_seg
                
                label_tmp = label_new_pad[ind_begin: ind_end]
                ecg_tmp = ecg_data_pad[ind_begin: ind_end]
                
                if ecg_data_mat is None:
                    ecg_data_mat = np.array(ecg_tmp)
                    label_new_mat = np.array(label_tmp)
                else:
                    ecg_data_mat = np.vstack((ecg_data_mat,ecg_tmp)) 
                    label_new_mat = np.vstack((label_new_mat,label_tmp))   
        else:
            break
            
        ind = ind_begin+len_seg
    """
    scio.savemat("ecgdata_"+FS+"_"dataid+".mat", {'Data':ecg_data_mat, 'Label':label_new_mat})
    plt.figure( )
    plt.plot(ecg_data_mat[2,])
    plt.plot(label_new_mat[2,])
    plt.show()
    """
    return ecg_data_mat, label_new_mat

def load_ans(is_load_train = True, is_select = 0):
    """
    Function for loading the detection results and references
    Input:
        is_load_train：是否加载训练数据
        is_select：是否只加载标签不为零的数据
    Ouput:
        S_refs: position references for S
        V_refs: position references for V
        S_results: position results for S
        V_results: position results for V
    """
    data_files = glob.glob(DATA_PATH + '*.mat')
    ref_files = glob.glob(REF_PATH + '*.mat')

    data_files.sort()
    ref_files.sort()

    S_refs = []
    V_refs = []
    S_results = []
    V_results = []
    
    data = None
    label = None
    for i, data_file in enumerate(data_files):
        if (is_load_train==1 and (i<7 or i==9)):         #A01~A07以及A10作为训练数据
        # if (is_load_train==1):         #A01~A07以及A10作为训练数据  
            pass
        elif(is_load_train==0 and (i==7 or i==8)):       #A08、A09作为测试数据
            pass
        else:
            continue 
        
        dataid = data_file.split('\\')[-1].split('.')[0]
        print(dataid)
        
        # load ecg file
        ecg_data_ori = sio.loadmat(data_file)['ecg'].squeeze()
        len_ecg_ori = len(ecg_data_ori)

        # 数据下采样
        #ecg_data = signal.resample(ecg_data_ori, int(len_ecg_ori*CONST))
        ecg_data = ecg_data_ori[1::2]
        len_ecg = len(ecg_data)
        
        #数据预处理
        # ecg_data = med_filt_1d(ecg_data)
        ##ecg_data = butter_bandpass_forward_backward_filter(ecg_data, 0.5, 35, FS, order=5)
       
        # load answers
        s_ref = sio.loadmat(ref_files[i])['ref']['S_ref'][0, 0].squeeze()
        v_ref = sio.loadmat(ref_files[i])['ref']['V_ref'][0, 0].squeeze()
   
        #标签下采样
        s_ref = (s_ref*CONST).astype(int)
        v_ref = (v_ref*CONST).astype(int)

        label_new = [0]*len_ecg

        #数据标记
        firstLabel(label_new, s_ref, len_ecg, 1)
        firstLabel(label_new, v_ref, len_ecg, 2)
        
        ecg_data_split = []
        ecg_label_split = []
        label_new = np.array(label_new)
        
        for i in tqdm(range(int(ecg_data.shape[0]/len_seg))):
            index_begin = i*len_seg
            index_end = (i+1)*len_seg
            #ecg_seg = ecg_data[i*len_seg:(i+1)*len_seg]
            label_seg = label_new[index_begin:index_end]
            if (label_seg != 0).any() and is_select:
                if (label_seg[:FS] != 0 ).any():
                    index_begin = index_begin - FS
                elif (label_seg[-FS:] != 0 ).any():
                    index_begin = index_begin + FS
                else:
                    pass
                index_end = index_begin + len_seg
                
                ecg_data_split.append(ecg_data[index_begin:index_end].tolist())
                ecg_label_split.append(label_new[index_begin:index_end].tolist())
            elif i%3==0 and is_select:
            # elif i%2==0 and is_select:
                ecg_data_split.append(ecg_data[index_begin:index_end].tolist())
                ecg_label_split.append(label_new[index_begin:index_end].tolist())
            elif is_select==0:
                ecg_data_split.append(ecg_data[index_begin:index_end].tolist())
                ecg_label_split.append(label_new[index_begin:index_end].tolist())
            
        ecg_data_split = np.array(ecg_data_split)
        ecg_label_split = np.array(ecg_label_split)
        print("ecg data :",ecg_data_split.shape)
        print("ecg label :",ecg_label_split.shape)
        
#         ecg_data_split = []
        
#         for i in tqdm(range(int(ecg_data.shape[0]/len_seg))):
#             ecg_data_split.append(ecg_data[i*len_seg:(i+1)*len_seg].tolist())
#         ecg_data_split = np.array(ecg_data_split)
        
#         ecg_label_split = []
        
#         label_new = np.array(label_new)
#         for i in tqdm(range(int(label_new.shape[0]/len_seg))):
#             ecg_label_split.append(label_new[i*len_seg:(i+1)*len_seg].tolist())
#         ecg_label_split = np.array(ecg_label_split)

        # 保存分段后数据
        #ecg_data_mat, label_new_mat = resave_data(ecg_data, label_new, dataid, is_select)
        
        if (ecg_data_split.shape[0] >0):
            if data is None:
                data  = ecg_data_split   
                label = ecg_label_split 
            else:
                data  = np.vstack((data, ecg_data_split))
                label = np.vstack((label, ecg_label_split))
    
    # try:
    #     data = getSigArr(data.T).T
    # except:
    #     pass
    
    return data, label

from keras import backend as K
def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
    
def f1(y_true, y_pred):
    precisionVal = precision(y_true, y_pred)
    recallVal = recall(y_true, y_pred)
    return 2*((precisionVal*recallVal)/(precisionVal+recallVal+K.epsilon()))

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

#写一个LossHistory类，保存loss和acc  
class LossHistory(keras.callbacks.Callback):  
    def on_train_begin(self, logs={}):  
        self.losses = {'batch':[], 'epoch':[]}  
        self.accuracy = {'batch':[], 'epoch':[]}  
        self.val_loss = {'batch':[], 'epoch':[]}  
        self.val_acc = {'batch':[], 'epoch':[]}  

    def on_batch_end(self, batch, logs={}):  
        self.losses['batch'].append(logs.get('loss'))  
        self.accuracy['batch'].append(logs.get('acc'))  
        self.val_loss['batch'].append(logs.get('val_loss'))  
        self.val_acc['batch'].append(logs.get('val_acc'))  

    def on_epoch_end(self, batch, logs={}):  
        self.losses['epoch'].append(logs.get('loss'))  
        self.accuracy['epoch'].append(logs.get('acc'))  
        self.val_loss['epoch'].append(logs.get('val_loss'))  
        self.val_acc['epoch'].append(logs.get('val_acc'))  

    def loss_plot(self, loss_type):  
        iters = range(len(self.losses[loss_type]))  
        plt.figure()  
        # acc  
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')  
        # loss  
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')  
        if loss_type == 'epoch':  
            # val_acc  
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')  
#             val_loss  
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')  
        plt.grid(True)  
        plt.xlabel(loss_type)  
        plt.ylabel('acc-loss')  
        plt.legend(loc="upper right")  
        plt.show()  
from keras.models import Model, load_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,core
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D, AveragePooling1D
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.core import Dense, Lambda
from keras.layers.core import Activation
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.layers import LeakyReLU
ACTIVATION = "relu"
def __grouped_convolution_block(blockInput, grouped_channels, cardinality, strides, weight_decay=5e-4,filter_size=15):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''
    init = blockInput
    group_list = []
    
    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv1D(grouped_channels, filter_size, padding='same', use_bias=False, strides=(strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
#         x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
#         if K.image_data_format() == 'channels_last' else
#         lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)
        x =  Lambda(lambda z: z[:, :, c * grouped_channels:(c + 1) * grouped_channels])(blockInput)
    
        x = Conv1D(grouped_channels, filter_size, padding='same', use_bias=False, strides=(strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

        group_list.append(x)

    group_merge = concatenate(group_list)#axis=channel_axis
    x = BatchNormalization()(group_merge)
    x = Activation('relu')(x)

    return x

def resnext_bottleneck_block(blockInput, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    ''' Adds a bottleneck block
    Args:
        input: input tensor
        filters: number of output filters
        cardinality: cardinality factor described number of
            grouped convolutions
        strides: performs strided convolution for downsampling if > 1
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    init = blockInput

    grouped_channels = int(filters / cardinality)
#     channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

#     # Check if input number of filters is same as 16 * k, else create convolution2d for this input
#     if K.image_data_format() == 'channels_first':
#         if init._keras_shape[1] != 2 * filters:
#             init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
#                           use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
#             init = BatchNormalization(axis=channel_axis)(init)
#     else:
#         if init._keras_shape[-1] != 2 * filters:
#             init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
#                           use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
#             init = BatchNormalization(axis=channel_axis)(init)

    init = Conv1D(filters * 2, 1, padding='same', strides=(strides),
                  use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
    init = BatchNormalization()(init)
    
    x = Conv1D(filters, 1, padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(blockInput)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = Conv1D(filters * 2, 1, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)

    x = add([init, x])
    x = Activation('relu')(x)

    return x

def convolution_block(x, filters, filter_size, strides=1, padding='same', activation=True):
    x = Conv1D(filters, filter_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = Activation(ACTIVATION)(x)
    return x

def residual_block(blockInput, num_filters=16,filter_size=15):
    x = Activation(ACTIVATION)(blockInput)
    x = BatchNormalization()(x)
    x = convolution_block(x, num_filters, filter_size)
    x = convolution_block(x, num_filters, filter_size, activation=False)
    x = Add()([x, blockInput])
    return x

def bottleneck(blockInput, num_filters=16, block="resnet"):
    if block == "resnet":
        conv = residual_block(blockInput,num_filters)
        conv = residual_block(blockInput,num_filters)
    elif block == "resnext":
        conv = resnext_bottleneck_block(blockInput,num_filters)
    return conv

def headneck(blockInput, num_neurons=2, block="resnet"):
    conv = []
    conv.append(Conv1D(num_neurons, 3, activation=None, padding="same")(blockInput))
    conv.append(Conv1D(num_neurons, 5, activation=None, padding="same")(blockInput))
    conv.append(Conv1D(num_neurons, 7, activation=None, padding="same")(blockInput))
    conv.append(Conv1D(num_neurons, 11, activation=None, padding="same")(blockInput))
    conv.append(Conv1D(num_neurons, 15, activation=None, padding="same")(blockInput))
    con_conv = concatenate(conv)
    
    return con_conv

# Build model
def build_model(input_layer, start_neurons, block="resnet", DropoutRatio = 0.5, filter_size=15, nClasses=2):
    # 101 -> 50
    
    # conv1 = headneck(input_layer)
    # conv1 = Conv1D(start_neurons * 1, filter_size, activation=None, padding="same")(conv1)
    
    conv1 = Conv1D(start_neurons * 1, filter_size, activation=None, padding="same")(input_layer)
    
    conv1 = bottleneck(conv1,start_neurons * 1, block)
    
    conv1 = Activation(ACTIVATION)(conv1)
    pool1 = MaxPooling1D((2))(conv1)
    pool1 = Dropout(DropoutRatio/2)(pool1)

    # 50 -> 25
    conv2 = Conv1D(start_neurons * 2, filter_size, activation=None, padding="same")(pool1)
    
    conv2 = bottleneck(conv2,start_neurons * 2, block)
    
    conv2 = Activation(ACTIVATION)(conv2)
    pool2 = MaxPooling1D((2))(conv2)
    pool2 = Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = Conv1D(start_neurons * 4, filter_size, activation=None, padding="same")(pool2)

    conv3 = bottleneck(conv3,start_neurons * 4, block)
    
    conv3 = Activation(ACTIVATION)(conv3)
    pool3 = MaxPooling1D((2))(conv3)
    pool3 = Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = Conv1D(start_neurons * 8, filter_size, activation=None, padding="same")(pool3)
#     conv4 = residual_block(conv4,start_neurons * 8)
#     conv4 = residual_block(conv4,start_neurons * 8)
    conv4 = bottleneck(conv4,start_neurons * 8, block)
    
    conv4 = Activation(ACTIVATION)(conv4)
    pool4 = MaxPooling1D((2))(conv4)
    pool4 = Dropout(DropoutRatio)(pool4)

    # Middle
    convm = Conv1D(start_neurons * 16, filter_size, activation=None, padding="same")(pool4)
#     convm = residual_block(convm,start_neurons * 16)
#     convm = residual_block(convm,start_neurons * 16)
    convm = bottleneck(convm,start_neurons * 16, block)
    
    convm = Activation(ACTIVATION)(convm)
    
    # 6 -> 12
    #deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    deconv4 = Conv1D(start_neurons * 8, filter_size,activation='relu', padding='same'
                     )(UpSampling1D(size=2)(convm))#kernel_initializer='he_normal'
    
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(DropoutRatio)(uconv4)
    
    uconv4 = Conv1D(start_neurons * 8, filter_size, activation=None, padding="same")(uconv4)
#     uconv4 = residual_block(uconv4,start_neurons * 8)
#     uconv4 = residual_block(uconv4,start_neurons * 8)
    uconv4 = bottleneck(uconv4,start_neurons * 8, block)
    
    uconv4 = Activation(ACTIVATION)(uconv4)
    
    # 12 -> 25
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    deconv3 = Conv1D(start_neurons * 4, filter_size, activation='relu', padding='same',
                     )(UpSampling1D(size=2)(uconv4))#kernel_initializer='he_normal'
    uconv3 = concatenate([deconv3, conv3])    
    uconv3 = Dropout(DropoutRatio)(uconv3)
    
    uconv3 = Conv1D(start_neurons * 4, filter_size, activation=None, padding="same")(uconv3)
#     uconv3 = residual_block(uconv3,start_neurons * 4)
#     uconv3 = residual_block(uconv3,start_neurons * 4)
    uconv3 = bottleneck(uconv3,start_neurons * 4, block)

    uconv3 = Activation(ACTIVATION)(uconv3)

    # 25 -> 50
    #deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    deconv2 = Conv1D(start_neurons * 2, filter_size, activation='relu', padding='same',
                     )(UpSampling1D(size=2)(uconv3))#kernel_initializer='he_normal'
    uconv2 = concatenate([deconv2, conv2])
        
    uconv2 = Dropout(DropoutRatio)(uconv2)
    uconv2 = Conv1D(start_neurons * 2, filter_size, activation=None, padding="same")(uconv2)
#     uconv2 = residual_block(uconv2,start_neurons * 2)
#     uconv2 = residual_block(uconv2,start_neurons * 2)
    uconv2 = bottleneck(uconv2,start_neurons * 2, block)

    uconv2 = Activation(ACTIVATION)(uconv2)
    
    # 50 -> 101
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    deconv1 = Conv1D(start_neurons * 1, filter_size, activation='relu', padding='same',
                     )(UpSampling1D(size=2)(uconv2))#kernel_initializer='he_normal'
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(DropoutRatio)(uconv1)
    uconv1 = Conv1D(start_neurons * 1, filter_size, activation=None, padding="same")(uconv1)
#     uconv1 = residual_block(uconv1,start_neurons * 1)
#     uconv1 = residual_block(uconv1,start_neurons * 1)
    uconv1 = bottleneck(uconv1,start_neurons * 1, block)
    
    uconv1 = Activation(ACTIVATION)(uconv1)
    
    uconv1 = Dropout(DropoutRatio/2)(uconv1)
    #output_layer = Conv1D(1, 1, padding="same", activation="sigmoid")(uconv1)
    output_layer = Conv1D(nClasses, 1, activation='relu', padding='same')(uconv1)#kernel_initializer='he_normal'
    #output_layer = core.Reshape((nClasses, input_length))(output_layer)
    #output_layer = core.Permute((2, 1))(output_layer)
    output_layer = core.Activation('softmax')(output_layer)
    #model = Model(inputs=inputs, outputs=conv9)
    
    return output_layer
# NestUnet
from keras.layers import LeakyReLU
deep_supervision = 0
debug = 0

nb_filter = [16,32,64,128,256]

act = "relu"

def cov_layer(x, filter_num, dropout,filter_size = 32, batch_norm=True):
    conv = Conv1D(filter_num, filter_size, padding='same', kernel_initializer='he_normal')(x)
    
    if batch_norm:
        conv = BatchNormalization()(conv)
        
    conv = Activation(act)(conv)    
    conv = Dropout(dropout)(conv)
        
    conv = Conv1D(filter_num, filter_size, padding='same', kernel_initializer='he_normal')(conv)
    
    if batch_norm:
        conv = BatchNormalization()(conv)
        
    conv = Activation(act)(conv)    
    conv = Dropout(dropout)(conv)
    
    res_conv = conv
    return res_conv

def NestUnet(nClasses, optimizer=None, input_length=len_ecg, filter_size = 32, nChannels=1):
    
    inputs = Input((input_length, nChannels))
    
    conv1_1 = cov_layer(inputs, nb_filter[0], 0)
    pool1 = MaxPooling1D(pool_size=2)(conv1_1)
    
    conv2_1 = cov_layer(pool1, nb_filter[1], 0.2)
    pool2 = MaxPooling1D(pool_size=2)(conv2_1)
    
    up1_2 = Conv1D(nb_filter[0], filter_size, padding='same')(UpSampling1D(size=2)(conv2_1))
    conv1_2 = concatenate([up1_2, conv1_1], axis=-1)
    conv1_2 = cov_layer(conv1_2, nb_filter[0], 0)
    
    conv3_1 = cov_layer(pool2, nb_filter[2], 0)
    pool3 = MaxPooling1D(pool_size=2)(conv3_1)

    up2_2 = Conv1D(nb_filter[1], filter_size, padding='same')(UpSampling1D(size=2)(conv3_1))
    conv2_2 = concatenate([up2_2, conv2_1], axis=-1)
    conv2_2 = cov_layer(conv2_2, nb_filter[1], 0.2)
    
    up1_3 = Conv1D(nb_filter[0], filter_size, padding='same')(UpSampling1D(size=2)(conv2_2))
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], axis=-1)
    conv1_3 = cov_layer(conv1_3, nb_filter[0], 0)
    
    conv4_1 = cov_layer(pool3, nb_filter[3], 0)
    pool4 = MaxPooling1D(pool_size=2)(conv4_1)
    
    up3_2 = Conv1D(nb_filter[2], filter_size, padding='same')(UpSampling1D(size=2)(conv4_1))
    conv3_2 = concatenate([up3_2, conv3_1], axis=-1)
    conv3_2 = cov_layer(conv3_2, nb_filter[2], 0)
    
    up2_3 = Conv1D(nb_filter[1], filter_size, padding='same')(UpSampling1D(size=2)(conv3_2))
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], axis=-1)
    conv2_3 = cov_layer(conv2_3, nb_filter[1], 0.2)
    
    up1_4 = Conv1D(nb_filter[0], filter_size, padding='same')(UpSampling1D(size=2)(conv2_3))
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], axis=-1)
    conv1_4 = cov_layer(conv1_4, nb_filter[0], 0)
    
    conv5_1 = cov_layer(pool4, nb_filter[4], 0)               #20200417修改：nb_filter[3]
    
    up4_2 = Conv1D(nb_filter[3], filter_size, padding='same')(UpSampling1D(size=2)(conv5_1))
    conv4_2 = concatenate([up4_2, conv4_1], axis=-1)
    conv4_2 = cov_layer(conv4_2, nb_filter[3], 0)
    
    up3_3 = Conv1D(nb_filter[2], filter_size, padding='same')(UpSampling1D(size=2)(conv4_2))
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], axis=-1)
    conv3_3 = cov_layer(conv3_3, nb_filter[2], 0.2)
    
    up2_4 = Conv1D(nb_filter[1], filter_size, padding='same')(UpSampling1D(size=2)(conv3_3))
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], axis=-1)
    conv2_4 = cov_layer(conv2_4, nb_filter[1], 0)
    
    up1_5 = Conv1D(nb_filter[0], filter_size, padding='same')(UpSampling1D(size=2)(conv2_4))
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], axis=-1)
    conv1_5 = cov_layer(conv1_5, nb_filter[0], 0)
    
#     x = Bidirectional(CuDNNGRU(32,return_sequences=True))(conv1_5)
#     x = Dropout(0.2)(x)
  
    if deep_supervision:
        nestnet_output_1 = Conv1D(1, 1, activation='sigmoid', kernel_initializer = 'he_normal', padding='same')(conv1_2)
        nestnet_output_2 = Conv1D(1, 1, activation='sigmoid', kernel_initializer = 'he_normal', padding='same')(conv1_3)
        nestnet_output_3 = Conv1D(1, 1, activation='sigmoid', kernel_initializer = 'he_normal', padding='same')(conv1_4)
        nestnet_output_4 = Conv1D(1, 1, activation='sigmoid', kernel_initializer = 'he_normal', padding='same')(conv1_5)
    
        model = Model(input=inputs, output=[nestnet_output_1,
                                            nestnet_output_2,
                                            nestnet_output_3,
                                            nestnet_output_4])
    else:    
#         nestnet_output = Conv1D(1, 1, activation='sigmoid', kernel_initializer = 'he_normal', padding='same')(conv1_5)
#         model = Model(input=inputs, output=nestnet_output)
        
        conv6 = Conv1D(nClasses, 1, activation='relu', kernel_initializer = 'he_normal', padding='same')(conv1_5)
        #conv6 = core.Reshape((nClasses, input_length))(conv6)
        #conv6 = core.Permute((2, 1))(conv6)
        
        nestnet_output = core.Activation('softmax')(conv6)  #sigmoid
        
        model = Model(input=inputs, output=nestnet_output)
        
    return model
# model = NestUnet(3)
# model.summary()
# Custom loss function
# def dice_coef(y_true, y_pred):
#     smooth = 1.
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
    
# #     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#     return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)

# def dice_coef_loss(y_true, y_pred):
#     return 1. - dice_coef(y_true, y_pred)

# def bce_dice_loss(y_true, y_pred):
#     return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    axis_to_reduce = [1,2]#range(1, K.ndim(y_pred))  # Reduce all axis but first (batch)
    numerator = y_true * y_pred #* class_weights  # Broadcasting
    numerator = 2. * K.sum(numerator, axis=axis_to_reduce)

    denominator = (y_true + y_pred) #* class_weights # Broadcasting
    denominator = K.sum(denominator, axis=axis_to_reduce)

    return 1 - numerator / denominator

def ce_dice_loss(y_true, y_pred):
    return  keras.losses.categorical_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)
def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)

def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
    y_true: the expected y values as a one-hot
    y_pred: the predicted y values as a one-hot or softmax output
    label: the label to return the IoU for
    Returns:
    the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)
 
def mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
    y_true: the expected y values as a one-hot
    y_pred: the predicted y values as a one-hot or softmax output
    Returns:
    the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1] - 1
    # initialize a variable to store total IoU in
    mean_iou = K.variable(0)

    # iterate over labels to calculate IoU for
    for label in range(1,num_labels+1):
        mean_iou = mean_iou + iou(y_true, y_pred, label)

    # divide total IoU by number of labels to get mean IoU
    return mean_iou / num_labels

def spb_iou(y_true, y_pred):
    mean_iou = K.variable(0)
    mean_iou = iou(y_true, y_pred, 1)
    return mean_iou

def pvc_iou(y_true, y_pred):
    mean_iou = K.variable(0)
    mean_iou = iou(y_true, y_pred, 2)
    return mean_iou


import pywt
def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1))
    return X * scalingFactor

def shift(ecg, interval=1):
    offset = np.random.normal(loc=1.0, scale=0.1, size=(1))
    #np.random.choice(range(-interval, interval))
    return ecg + offset[0]

def wavelet_db4(ecg, wavefunc='db4', lv=4, m=2, n=4):  #

    coeff = pywt.wavedec(ecg, wavefunc, mode='sym', level=lv)  #
    # sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0

    for i in range(m, n + 1):
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2 * np.log(len(cD)))
            if cD[j] >= Tr:
                coeff[i][j] = np.sign(cD[j]) - Tr
            else:
                coeff[i][j] = 0

    denoised_ecg = pywt.waverec(coeff, wavefunc)
    return denoised_ecg

def wavelet_db6(sig,wavefunc='db6',level=6):
    """
    R J, Acharya U R, Min L C. ECG beat classification using PCA, LDA, ICA and discrete
     wavelet transform[J].Biomedical Signal Processing and Control, 2013, 8(5): 437-448.
    param sig: 1-D numpy Array
    return: 1-D numpy Array
    """
    coeffs = pywt.wavedec(sig, wavefunc, level=level)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, wavefunc)
    return sig_filt

def wavelet_sym(sig,wavefunc='sym6',level=6):
    """
    R J, Acharya U R, Min L C. ECG beat classification using PCA, LDA, ICA and discrete
     wavelet transform[J].Biomedical Signal Processing and Control, 2013, 8(5): 437-448.
    param sig: 1-D numpy Array
    return: 1-D numpy Array
    """
    coeffs = pywt.wavedec(sig, wavefunc, level=level)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, wavefunc)
    return sig_filt

class DataGenerator(keras.utils.Sequence):
    # ' Generates data for Keras '

    def __init__(self, data, list_Inds, labels, train=True, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=3, shuffle=True):
        # 'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_inds = list_Inds
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data = data
        self.train = train
        self.noise = np.load("noise.npy",allow_pickle=True).item()
        self.on_epoch_end()

    def __len__(self):
        # 'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_inds) / self.batch_size))

    def __getitem__(self, index):
        # 'Generate one batch of data'
        # Generate indexes of the batch
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        indexes = self.list_inds[index*self.batch_size:(index+1)*self.batch_size]
                
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        #self.indexes = np.arange(len(self.list_IDs))
        #if self.shuffle == True:
        #    np.random.shuffle(self.indexes)
        if self.shuffle == True:
            np.random.shuffle(self.list_inds)
            #if (self.list_inds == 27).any():
            #    print("Wrong")
                

    def __data_generation(self, list_inds):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        #X = np.empty((self.batch_size,  *self.dim, self.n_channels))
        #y = np.empty((self.batch_size, self.n_classes), dtype=int)
        #print(list_inds)
        
        X = self.data[list_inds]
        for i in range(X.shape[0]):
            sig = X[i,:,0]
            # # 数据增强
            if self.train:
                ind = np.random.randint(360)
                if np.random.randn() > 0.2:
                    if np.random.randn() > 0.5: sig = sig+self.noise['bw'][ind]*0.5
                    if np.random.randn() > 0.5: sig = sig+self.noise['em'][ind]*0.2
                    if np.random.randn() > 0.5: sig = sig+self.noise['ma'][ind]*0.2
                else:
                    pass

                # if np.random.randn() > 0.5: sig = scaling(sig)
                # elif np.random.randn() > 0.5: sig = shift(sig)
                # elif np.random.randn() > 0.3: sig = wavelet_db4(sig)
                # elif np.random.randn() > 0.3: sig = wavelet_db6(sig)
                # elif np.random.randn() > 0.3: sig = wavelet_sym(sig)
                    
                # if np.random.randn() > 0.5: sig = butter_bandpass_forward_backward_filter(sig,0.05,40,200)
                # #     fi = np.random.randint(11)
                # #     if fi % 2 == 0 and fi != 2 and fi != 0 :
                # #         sig = wavelet_db6(sig,'db{}'.format(fi) ,8)
                # #     else:#if  fi % 2 != 0:
                # #         if np.random.randn() > -0.5:
                # #             sig = butter_bandpass_filter(sig,0.05,40,256)
                # #         else:
                # #             sig = butter_bandpass_forward_backward_filter(sig,0.05,40,256)
                X[i,:,0] = sig
            else:
                # sig = butter_bandpass_filter(sig,0.05,40,256)
                pass

        X = np.expand_dims(getSigArr(X[:,:,0]), axis=2)

        y = self.labels[list_inds]

        # Generate data
        y = to_categorical(y, num_classes=self.n_classes)

        return X, y  # keras.utils.to_categorical(y, num_classes=self.n_classes)

data_train, label_train = load_ans(is_load_train = 1, is_select = 1)
# data_test, label_test = load_ans(False,0)

# x_test = np.expand_dims(data_test, axis=2)
# y_test = np.expand_dims(label_test, axis=2)

x_train = np.expand_dims(data_train, axis=2)
y_train = np.expand_dims(label_train, axis=2)


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=2020)

y_ind = np.zeros(len(label_train))
s = 0
v = 0
n = 0
for i in tqdm(range(len(label_train))):
    if (label_train[i] == 1).any():
        y_ind[i] = 1
        s += 1
    elif (label_train[i] == 2).any():
        y_ind[i] = 2
        v += 1
    else:
        n += 1
        
print(s,v,n)

for train_index, test_index in skf.split(label_train,y_ind):
    print(train_index, test_index)
    break
    
total_index = np.array(train_index.tolist()+test_index.tolist())

# training_generator = DataGenerator(x_train, train_index, y_train, **params)
# for i in training_generator:
#     print(i[1].shape)
#     break

# https://github.com/henyau/Image-Segmentation-with-Unet/blob/master/train.py
def dice(y_pred, y_true):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def fbeta(y_pred, y_true):

    pred0 = Lambda(lambda x : x[:,:,:,0])(y_pred)
    pred1 = Lambda(lambda x : x[:,:,:,1])(y_pred)
    true0 = Lambda(lambda x : x[:,:,:,0])(y_true)
    true1 = Lambda(lambda x : x[:,:,:,1])(y_true) # channel last?
    
    y_pred_0 = K.flatten(pred0)
    y_true_0 = K.flatten(true0)
    
    y_pred_1 = K.flatten(pred1)
    y_true_1 = K.flatten(true1)
    
    intersection0 = K.sum(y_true_0 * y_pred_0)
    intersection1 = K.sum(y_true_1 * y_pred_1)

    precision0 = intersection0/(K.sum(y_pred_0)+K.epsilon())
    recall0 = intersection0/(K.sum(y_true_0)+K.epsilon())
    
    precision1 = intersection1/(K.sum(y_pred_1)+K.epsilon())
    recall1 = intersection1/(K.sum(y_true_1)+K.epsilon())
    
    fbeta0 = (1.0+0.25)*(precision0*recall0)/(0.25*precision0+recall0+K.epsilon())
    fbeta1 = (1.0+4.0)*(precision1*recall1)/(4.0*precision1+recall1+K.epsilon())
    
    return ((fbeta0+fbeta1)/2.0)

def fbeta_loss(y_true, y_pred):
    return 1-fbeta(y_true, y_pred)

def dice_loss(y_true, y_pred):
    return 1-dice(y_true, y_pred)

def weighted_categorical_crossentropy(y_true, y_pred):    
    #weights = K.variable([0.5,2.0,0.0])
    weights = K.variable([0.2,0.4,0.4])
        
    # scale predictions so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss

def cat_dice_loss(y_true, y_pred):
#    return categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    #return weighted_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)+fbeta_loss(y_true, y_pred)
    return weighted_categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    #return weighted_categorical_crossentropy(y_true, y_pred) +fbeta_loss(y_true, y_pred)
    #return dice_loss(y_true, y_pred)

import time
import shutil
time_ori = time.strftime('%Y-%m-%d %X').split(" ")[0]
def save_epoch(epoch, types):
    if (epoch>=10 and epoch%10==0):   
    # if (epoch>=10):   
        shutil.copyfile('{}_model_weights_{}.h5'.format(types, time_ori),
                        '{}_model_weights_{}_epoch{}.h5'.format(types,time_ori,epoch))


import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)
# 设置session
KTF.set_session(sess)

from keras.losses import categorical_crossentropy
from keras_radam import RAdam
# from keras_lookahead import Lookahead
from keras.callbacks import LearningRateScheduler
from keras.models import model_from_yaml

M = 3 # number of snapshots
nb_epoch = T = 180 # number of epochs
alpha_zero = 0.0001 # initial learning rate

np.random.seed(0)

model_type = "ysnet" #"resnet"#"nest"#
import net
model = net.build_model(start_neurons=16, dropout_ratio=0, filter_size=1, nClasses=3)

epoch_print_callback = keras.callbacks.LambdaCallback(
    on_epoch_end =lambda epoch,logs: save_epoch(epoch,model_type) )

checkpoint = ModelCheckpoint(filepath='{}_model_weights_{}.h5'.format(model_type, time.strftime('%Y-%m-%d %X').split(" ")[0]),
                                 monitor= 'val_mean_iou', mode='max', verbose=1,
                                 save_best_only=True,save_weights_only=True)

earlystop = EarlyStopping(
            monitor='val_mean_iou',
            mode='max',
            patience=5
          )

reducelr = ReduceLROnPlateau(monitor='val_mean_iou', factor=0.1, verbose=1,mode='max',
                             patience=3, min_lr=0.000001)

# reduce_lr = LearningRateScheduler(scheduler3)

#创建一个实例history
history = LossHistory()

callback_lists = [earlystop, checkpoint, history, reducelr, epoch_print_callback]


# model = NestUnet(3)
# model = Attention_unet()

# yaml_string = model.to_yaml()
# model = model_from_yaml(yaml_string)

# input_length = len_ecg
# input_layer = Input((input_length, 1))
# output_layer = build_model(input_layer=input_layer,block=model_type,start_neurons=16, DropoutRatio=0.5, filter_size=15, nClasses=3)
# model = Model(input_layer, output_layer)

model.compile(loss=categorical_crossentropy,# cat_dice_loss,#,#custom_loss,#  # bce_dice_loss, binary_crossentropy, categorical_crossentropy，mse
              optimizer=RAdam(0.0001),  # RAdam,'mse'，Lookahead(RAdam(lr=0.0001)),bce_dice_loss，focal_tversky
              metrics=['accuracy', spb_iou, pvc_iou, mean_iou, auc, f1])  # mse、categorical_crossentropy

# if deep_supervision:
#     model.fit(x_train, [y_train,y_train,y_train, y_train],
#           epochs=100, batch_size=64, verbose=2,
#           callbacks= callback_lists
#          )
# else:
#     model.fit(x_train, target_train,
#           validation_split=0.1, 
#           epochs=30, batch_size=64, verbose=2,
#           callbacks= callback_lists
#          )

# Parameters
params = {'dim': (2000,1),
          'batch_size': 128,#64,
          'n_classes': 3,
          'n_channels': 1,
          'shuffle': True}

# Generators
training_generator = DataGenerator(x_train, train_index, y_train,train=True, **params)  
validation_generator = DataGenerator(x_train, test_index, y_train, train=False,**params)

history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              #use_multiprocessing=False,
                              epochs=50,
                              verbose=2,
                              callbacks=callback_lists)

# history.loss_plot('epoch')
def CPSC2020_challenge(ECG, fs):
    """   """
    # 数据下采样
    ecg_data = ECG[1::2]
    len_ecg = len(ecg_data)

    #数据预处理
    # ecg_data = med_filt_1d(ecg_data)
    ##ecg_data = butter_bandpass_forward_backward_filter(ecg_data, 0.5, 35, FS, order=5)

    ecg_data_split = []

    for i in tqdm(range(int(ecg_data.shape[0]/len_seg))):
        ecg_data_split.append(ecg_data[i*len_seg:(i+1)*len_seg].tolist())
    ecg_data_split = np.array(ecg_data_split)

    ecg_data_split = getSigArr(ecg_data_split.T).T

    ecg_data_split = np.expand_dims(ecg_data_split, axis=2)
    
#     from keras.models import model_from_yaml
#     fname_model = 'resnet_model.yaml'
#     yaml_file = open(fname_model, 'r')
#     loaded_model_yaml = yaml_file.read()
#     yaml_file.close()
#     model = model_from_yaml(loaded_model_yaml)

#     # 加载模型权重
#     fname_model_weigths = 'unetplus_deepsupervision_model_weights_2020-10-15.h5'
#     model.load_weights(fname_model_weigths)
    
    pred_test = model.predict(ecg_data_split)

    target_pred = np.argmax(pred_test, axis = 2)  
    
    y_pred = target_pred.reshape(-1)
    
    s_count = 0
    v_count = 0

    s_results = []
    v_results = []

    for i in range(0,len(y_pred)):
        if (y_pred[i]>1):
            #print("vvv")
            v_count = v_count+1
        elif (y_pred[i]>0):
            #print("sss")
            s_count = s_count+1
        else:
            if (s_count>=200*0.1):#40
                ind = i- round(s_count*0.3)
                s_results.append(ind)

            if (v_count>=200*0.1):#40
                ind = i- round(v_count*0.3)
                v_results.append(ind)

            s_count=0
            v_count=0
            
    S_pos = (np.array(s_results)/CONST).astype(int)
    V_pos = (np.array(v_results)/CONST).astype(int)
    
    return S_pos, V_pos

FS = 400
THR = 0.15
DATA_PATH = '../TrainingSet/data/'
REF_PATH = '../TrainingSet/ref/'
import glob
import numpy as np
import os
import scipy.io as sio

def load_ans():
    """
    Function for loading the detection results and references
    Input:

    Ouput:
        S_refs: position references for S
        V_refs: position references for V
        S_results: position results for S
        V_results: position results for V
    """
    data_files = glob.glob(DATA_PATH + '*.mat')
    ref_files = glob.glob(REF_PATH + '*.mat')
    
    data_files.sort()
    ref_files.sort()
    
    data_files = data_files[7:9]
    ref_files  = ref_files[7:9]

    S_refs = []
    V_refs = []
    S_results = []
    V_results = []
    for i, data_file in enumerate(data_files[:]):
        print(data_file,ref_files[i])
        # load ecg file
        ecg_data = sio.loadmat(data_file)['ecg'].squeeze()
        # load answers
        s_ref = sio.loadmat(ref_files[i])['ref']['S_ref'][0, 0].squeeze()
        v_ref = sio.loadmat(ref_files[i])['ref']['V_ref'][0, 0].squeeze().reshape(-1)
        # process ecg and conduct event detection using your algorithm
        s_pos, v_pos = CPSC2020_challenge(ecg_data, FS)
        S_refs.append(s_ref)
        V_refs.append(v_ref)
        S_results.append(s_pos)
        V_results.append(v_pos)

    return S_refs, V_refs, S_results, V_results

S_refs, V_refs, S_results, V_results = load_ans()



# def CPSC2020_score(S_refs, V_refs, S_results, V_results):
#     """
#     Score Function
#     Input:
#         S_refs, V_refs, S_results, V_results
#     Output:
#         Score1: score for S
#         Score2: score for V
#     """
#     s_score = np.zeros([len(S_refs), ])
#     v_score = np.zeros([len(S_refs), ])
#     ## Scoring ##
#     for i, s_ref in enumerate(S_refs):
#         v_ref = V_refs[i]
#         s_pos = S_results[i]
#         v_pos = V_results[i]
#         s_tp = 0
#         s_fp = 0
#         s_fn = 0
#         v_tp = 0
#         v_fp = 0
#         v_fn = 0
#         if s_ref.size == 0:
#             s_fp = len(s_pos)
#         else:
#             for m, ans in enumerate(s_ref):
#                 s_pos_cand = np.where(abs(s_pos-ans) <= THR*FS)[0]
#                 if s_pos_cand.size == 0:
#                     s_fn += 1
#                 else:
#                     s_tp += 1
#                     s_fp += len(s_pos_cand) - 1
#         if v_ref.size == 0:
#             v_fp = len(v_pos)
#         else:
#             for m, ans in enumerate(v_ref):
#                 v_pos_cand = np.where(abs(v_pos-ans) <= THR*FS)[0]
#                 if v_pos_cand.size == 0:
#                     v_fn += 1
#                 else:
#                     v_tp += 1
#                     v_fp += len(v_pos_cand) - 1
#         # calculate the score
#         s_score[i] = s_fp * (-1) + s_fn * (-5)
#         v_score[i] = v_fp * (-1) + v_fn * (-5)
#     Score1 = np.sum(s_score)
#     Score2 = np.sum(v_score)

#     return Score1, Score2

def CPSC2020_score(S_refs, V_refs, S_results, V_results):
    """
    Score Function
    Input:
        S_refs, V_refs, S_results, V_results
    Output:
        Score1: score for S
        Score2: score for V
    """
    s_score = np.zeros([len(S_refs), ])
    v_score = np.zeros([len(S_refs), ])
    ## Scoring ##
    for i, s_ref in enumerate(S_refs):
        v_ref = V_refs[i]
        s_pos = S_results[i]
        v_pos = V_results[i]
        s_tp = 0
        s_fp = 0
        s_fn = 0
        v_tp = 0
        v_fp = 0
        v_fn = 0
        if s_ref.size == 0:
            s_fp = len(s_pos)
        else:
            for m, ans in enumerate(s_ref):
                s_pos_cand = np.where(abs(s_pos-ans) <= THR*FS)[0]
                if s_pos_cand.size == 0:
                    s_fn += 1
                else:
                    s_tp += 1
            s_fp += (len(s_pos) - s_tp)

        if v_ref.size == 0:
            v_fp = len(v_pos)
        else:
            for m, ans in enumerate(v_ref):
                v_pos_cand = np.where(abs(v_pos-ans) <= THR*FS)[0]
                if v_pos_cand.size == 0:
                    v_fn += 1
                else:
                    v_tp += 1
            v_fp += (len(v_pos) - v_tp)

        # calculate the score
        s_score[i] = s_fp * (-1) + s_fn * (-5)
        v_score[i] = v_fp * (-1) + v_fn * (-5)
    Score1 = np.sum(s_score)
    Score2 = np.sum(v_score)

    return Score1, Score2
index = 11
S1, S2 = CPSC2020_score(S_refs[:index], V_refs[:index], S_results[:index], V_results[:index])

print ("S_score: {}".format(S1))
print ("V_score: {}".format(S2))









