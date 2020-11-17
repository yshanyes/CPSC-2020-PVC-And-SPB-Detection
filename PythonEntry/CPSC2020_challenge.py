import numpy as np
from keras.models import model_from_yaml
# from tqdm import tqdm
from sklearn import preprocessing as prep
import tensorflow as tf
import os
import keras.backend.tensorflow_backend as KTF
import resnet
import resnext

FS_ORI = 400
FS = 200
len_seg = 2000
CONST = 200/400

os.environ["CUDA_VISIBLE_DEVICES"] = '0'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.Session(config = config)
KTF.set_session(sess) # 设置session

# load model architecture
model_type = "resnet"
# resnet_model = resnet.build_model(block=model_type,start_neurons=16, DropoutRatio=0.5, filter_size=15, nClasses=3)
resnet_model = resnet.build_model(block=model_type,start_neurons=16, DropoutRatio=0.5, filter_size=32, nClasses=3)
# load weight
resnet_model.load_weights('resnet_model_weights.h5')

# load model architecture
model_type = "resnext"
resnext_model = resnext.build_model(block=model_type,start_neurons=16, DropoutRatio=0.5, filter_size=15, nClasses=3)

# load weight
resnext_model.load_weights('resnext_model_weights.h5')


# preprocessing
def getSigArr(sig, sigNorm='minmax'):
    if sigNorm == 'scale':
        sig = prep.scale(sig)
    elif sigNorm == 'minmax':
        min_max_scaler = prep.MinMaxScaler()
        sig = min_max_scaler.fit_transform(sig)
    return sig

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


def CPSC2020_challenge(ECG, fs):
    """
    % This function can be used for events 1 and 2. Participants are free to modify any
    % components of the code. However the function prototype must stay the same
    % [S_pos,V_pos] = CPSC2020_challenge(ECG,fs) where the inputs and outputs are specified
    % below.
    %
    %% Inputs
    %       ECG : raw ecg vector signal 1-D signal
    %       fs  : sampling rate
    %
    %% Outputs
    %       S_pos : the position where SPBs detected
    %       V_pos : the position where PVCs detected
    %
    %
    %
    % Copyright (C) 2020 Dr. Chengyu Liu
    % Southeast university
    % chengyu@seu.edu.cn
    %
    % Last updated : 02-23-2020

    """

#   ====== arrhythmias detection =======

#    S_pos = np.zeros([1, ])
#    V_pos = np.zeros([1, ])
    """   """
    # 
    ecg_data = ECG[1::2]
    len_ecg = len(ecg_data)

    #
    ecg_data = med_filt_1d(ecg_data)
    ##ecg_data = butter_bandpass_forward_backward_filter(ecg_data, 0.5, 35, FS, order=5)

    ecg_data_split = []

    for i in range(int(ecg_data.shape[0]/len_seg)):
        ecg_data_split.append(ecg_data[i*len_seg:(i+1)*len_seg].tolist())
    ecg_data_split = np.array(ecg_data_split)

    ecg_data_split = getSigArr(ecg_data_split.T).T

    ecg_data_split = np.expand_dims(ecg_data_split, axis=2)
    
    pred_test = resnet_model.predict(ecg_data_split)
    target_pred = np.argmax(pred_test, axis = 2)
    y_pred = target_pred.reshape(-1)
    
    s_count = 0
    v_count = 0
    s_results = []
    v_results = []

    for i in range(0,len(y_pred)):
        if (y_pred[i]>1):
            v_count = v_count+1
        elif (y_pred[i]>0):
            s_count = s_count+1
        else:
            if (s_count>=200*0.05):#40
                ind = i- round(s_count*0.5)
                s_results.append(ind)
            if (v_count>=200*0.05):#40
                ind = i- round(v_count*0.5)
                v_results.append(ind)
            s_count=0
            v_count=0
            
    S_pos = (np.array(s_results)/CONST).astype(int)
    # V_pos = (np.array(v_results)/CONST).astype(int)


    ecg_data = ECG[1::2]
    len_ecg = len(ecg_data)
    ecg_data_split = []
    for i in range(int(ecg_data.shape[0]/len_seg)):
        ecg_data_split.append(ecg_data[i*len_seg:(i+1)*len_seg].tolist())
    ecg_data_split = np.array(ecg_data_split)
    ecg_data_split = getSigArr(ecg_data_split.T).T
    ecg_data_split = np.expand_dims(ecg_data_split, axis=2)

    pred_test = resnext_model.predict(ecg_data_split)
    target_pred = np.argmax(pred_test, axis = 2)
    y_pred = target_pred.reshape(-1)
    
    s_count = 0
    v_count = 0
    s_results = []
    v_results = []

    for i in range(0,len(y_pred)):
        if (y_pred[i]>1):
            v_count = v_count+1
        elif (y_pred[i]>0):
            s_count = s_count+1
        else:
            if (s_count>=200*0.1):#40
                ind = i- round(s_count*0.5)
                s_results.append(ind)
            if (v_count>=200*0.1):#40
                ind = i- round(v_count*0.5)
                v_results.append(ind)
            s_count=0
            v_count=0
            
    # S_pos = (np.array(s_results)/CONST).astype(int)
    V_pos = (np.array(v_results)/CONST).astype(int)


    return S_pos, V_pos
