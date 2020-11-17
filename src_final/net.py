from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, LeakyReLU
from keras.layers import Dense, Dropout, Activation, Flatten,  Input, Reshape, GRU, CuDNNGRU,CuDNNLSTM
from keras.layers import Convolution1D, MaxPool1D, GlobalAveragePooling1D,concatenate,AveragePooling1D,GlobalMaxPooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints
from keras.layers import Layer
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import Reshape

from keras.layers import Input,Dropout,BatchNormalization,Activation,Add,core,Multiply
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D, AveragePooling1D
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.core import Dense, Lambda
from keras.layers.core import Activation
from keras.layers import Input
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K
from keras.layers import LeakyReLU

# https://github.com/nibtehaz/MultiResUNet/blob/master/MultiResUNet.py
# https://github.com/ybabakhin/kaggle_salt_bes_phalanx/blob/master/phalanx/unet_model.py

def dot_product(x, kernel):
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def __grouped_convolution_block(blockInput, grouped_channels, cardinality, strides, weight_decay=5e-4,filter_size=5):
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
        x = Conv1D(grouped_channels, filter_size, dilation=2, padding='same', use_bias=False, strides=(strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
#         x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
#         if K.image_data_format() == 'channels_last' else
#         lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)
        x =  Lambda(lambda z: z[:, :, c * grouped_channels:(c + 1) * grouped_channels])(blockInput)
    
        x = Conv1D(grouped_channels, filter_size, dilation=2, padding='same', use_bias=False, strides=(strides),
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
    
#two parameters: input and reduction ratio
def squeeze_excite_block(block_input, ratio=8):
    filter_kernels = block_input._keras_shape[-1]
    z_shape = (1, filter_kernels)
    #z = GlobalAveragePooling1D()(block_input)
    # z = GlobalMaxPooling1D()(block_input)
    z = AttentionWithContext()(block_input)
    z = Reshape(z_shape)(z)
    s = Dense(filter_kernels//ratio, activation='relu', use_bias=False)(z)
    s = Dense(filter_kernels, activation='sigmoid', use_bias=False)(s)
    x = Multiply()([block_input, s])#multiply
    return x

def conv1d_bn(x, filters, filter_size, padding='same', dilation=1, strides=1, activation='relu', name=None):

    x = Conv1D(filters, filter_size, dilation_rate=dilation, strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    if(activation == None):
        return x
    x = Activation(activation, name=name)(x)
    return x

def MultiResBlock(U, inp, alpha = 1):
    '''
    MultiRes Block
    Arguments:
        U {int} -- Number of filters in a corrsponding UNet stage
        inp {keras layer} -- input layer 
    Returns:
        [keras layer] -- [output layer]
    '''
    W = alpha * U
    shortcut = inp

    shortcut = conv1d_bn(shortcut, int(W*0.167) + int(W*0.333) + int(W*0.5), 1, dilation=1, activation=None, padding='same')

    #print(int(W*0.167) + int(W*0.333) + int(W*0.5))

    conv3x3 = conv1d_bn(inp, int(W*0.167), 3,  dilation=2, activation='relu', padding='same')

    conv5x5 = conv1d_bn(conv3x3, int(W*0.333), 3, dilation=2, activation='relu', padding='same')

    conv7x7 = conv1d_bn(conv5x5, int(W*0.5), 3, dilation=2, activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5, conv7x7])
    out = BatchNormalization()(out)

    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization()(out)

    return out

def ResPath(inp, filters, length):
    '''
    ResPath
    Arguments:
        filters {int} -- [description]
        length {int} -- length of ResPath
        inp {keras layer} -- input layer 
    
    Returns:
        [keras layer] -- [output layer]
    '''
    shortcut = inp
    shortcut = conv1d_bn(shortcut, filters, 1, dilation=1, activation=None, padding='same')
    out = conv1d_bn(inp, filters, 3, dilation=2, activation='relu', padding='same')
    out = add([shortcut, out])
    out = Activation('relu')(out)
    out = BatchNormalization()(out)

    for i in range(length-1):
        shortcut = out
        shortcut = conv1d_bn(shortcut, filters, 1, dilation=1, activation=None, padding='same')
        out = conv1d_bn(out, filters, 3, dilation=2, activation='relu', padding='same')
        out = add([shortcut, out])
        out = Activation('relu')(out)
        out = BatchNormalization()(out)
    return out

def resnet_bottleneck(block_input,num_neurons,kernel_size):
    
    x = Conv1D(num_neurons, 1, padding='same', kernel_initializer='he_normal',use_bias=False)(block_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv1D(num_neurons, kernel_size, padding='same', dilation_rate=2, kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
        
    x = Conv1D(num_neurons, 1, padding='same', kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = squeeze_excite_block(x)
    out = add([x, block_input])
    

    return out

def convolution_block(x, filters, filter_size, strides=1, padding='same', activation='relu'):
    x = Conv1D(filters, filter_size, dilation_rate=2, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x

def residual_block(blockInput, num_filters=16, filter_size=5,activation='relu'):
    x = Activation(activation)(blockInput)
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

def se_bottleneck(block_input,num_neurons,kernel_size=[3,5,7]):

    x = Conv1D(num_neurons, 1, activation=None, padding="same")(block_input)
    x = bottleneck(x,num_neurons)
    
    # x = MultiResBlock(U=num_neurons, inp=block_input)
    x = Bidirectional(CuDNNLSTM(x._keras_shape[-1]//2+1, input_shape=x._keras_shape,return_sequences=True,return_state=False))(x)

    # x = Bidirectional(CuDNNGRU(x._keras_shape[-1]//2, input_shape=x._keras_shape,return_sequences=True,return_state=False))(x)
    # x = squeeze_excite_block(x)
    return x

def center_bottleneck(block_input,num_neurons,kernel_size=[3,5,7]):

    x = Conv1D(num_neurons, 1, activation=None, padding="same")(block_input)
    x = bottleneck(x,num_neurons)
    
    # x = MultiResBlock(U=num_neurons, inp=block_input)
    x = Bidirectional(CuDNNLSTM(x._keras_shape[-1]//2+1, input_shape=x._keras_shape,return_sequences=True,return_state=False))(x)

    x = Bidirectional(CuDNNGRU(x._keras_shape[-1]//2, input_shape=x._keras_shape,return_sequences=True,return_state=False))(x)
    x = squeeze_excite_block(x)
    return x

# Build model
def build_model(start_neurons=16, dropout_ratio = None, filter_size=1, nClasses=3, resPath=False):
    # 101 -> 50
    input_layer = Input(shape=(2000,1), dtype='float32', name='main_input')
       
    # conv1 = Conv1D(start_neurons * 1, filter_size, activation=None, padding="same")(input_layer)
    #conv1 = resnet_bottleneck(bottleneck,start_neurons * 1,3)

    conv1 = input_layer
    conv1 = se_bottleneck(conv1,start_neurons * 1)
    pool1 = MaxPooling1D((2))(conv1)
    if resPath:
        conv1 = ResPath(conv1,start_neurons * 1, 4)
    
    if dropout_ratio:
        pool1 = Dropout(dropout_ratio)(pool1)

    # 50 -> 25
    # conv2 = Conv1D(start_neurons * 2, filter_size, activation=None, padding="same")(pool1)
    conv2 = pool1
    conv2 = se_bottleneck(conv2,start_neurons * 2)
    pool2 = MaxPooling1D((2))(conv2)
    if resPath:
        conv2 = ResPath(conv2,start_neurons * 1, 3)
    
    if dropout_ratio:
        pool2 = Dropout(dropout_ratio)(pool2)

    # 25 -> 12
    # conv3 = Conv1D(start_neurons * 4, filter_size, activation=None, padding="same")(pool2)
    conv3 = pool2
    conv3 = se_bottleneck(conv3,start_neurons * 4)
    pool3 = MaxPooling1D((2))(conv3)
    if resPath:
        conv3 = ResPath(conv3,start_neurons * 1, 2)

    if dropout_ratio:
        pool3 = Dropout(dropout_ratio)(pool3)

    # 12 -> 6
    # conv4 = Conv1D(start_neurons * 8, filter_size, activation=None, padding="same")(pool3)
    conv4 = pool3
    conv4 = se_bottleneck(pool3,start_neurons * 8)
    pool4 = MaxPooling1D((2))(conv4)
    if resPath:
        conv4 = ResPath(conv4,start_neurons * 1, 1)

    if dropout_ratio:
        pool4 = Dropout(dropout_ratio)(pool4)

    # Middle
    # convm = Conv1D(start_neurons * 16, filter_size, activation=None, padding="same")(pool4)
    convm = pool4
    convm = center_bottleneck(convm,start_neurons * 16)#se_bottleneck(convm,start_neurons * 16)
    
    # 6 -> 12
    # deconv4 = Conv1D(start_neurons * 8, filter_size,activation='relu', padding='same'
    #                  )(UpSampling1D(size=2)(convm))#kernel_initializer='he_normal'
    deconv4 = UpSampling1D(size=2)(convm)
    uconv4 = concatenate([deconv4, conv4])
    
    if dropout_ratio:
        uconv4 = Dropout(dropout_ratio)(uconv4)
    #uconv4 = Conv1D(start_neurons * 8, filter_size, activation=None, padding="same")(uconv4)
    uconv4 = se_bottleneck(uconv4,start_neurons * 8)
        
    # 12 -> 25
    # deconv3 = Conv1D(start_neurons * 4, filter_size, activation='relu', padding='same',
    #                  )(UpSampling1D(size=2)(uconv4))#kernel_initializer='he_normal'
    deconv3 = UpSampling1D(size=2)(uconv4)      
    uconv3 = concatenate([deconv3, conv3]) 
    
    if dropout_ratio:
        uconv3 = Dropout(dropout_ratio)(uconv3)
    #uconv3 = Conv1D(start_neurons * 4, filter_size, activation=None, padding="same")(uconv3)
    uconv3 = se_bottleneck(uconv3,start_neurons * 4)

    # 25 -> 50
    # deconv2 = Conv1D(start_neurons * 2, filter_size, activation='relu', padding='same',
    #                  )(UpSampling1D(size=2)(uconv3))#kernel_initializer='he_normal'
    deconv2 = UpSampling1D(size=2)(uconv3)
    uconv2 = concatenate([deconv2, conv2])
        
    if dropout_ratio:
        uconv2 = Dropout(dropout_ratio)(uconv2)
    #uconv2 = Conv1D(start_neurons * 2, filter_size, activation=None, padding="same")(uconv2)
    uconv2 = se_bottleneck(uconv2,start_neurons * 2)
   
    # 50 -> 101
    # deconv1 = Conv1D(start_neurons * 1, filter_size, activation='relu', padding='same',
    #                  )(UpSampling1D(size=2)(uconv2))#kernel_initializer='he_normal'
    deconv1 = UpSampling1D(size=2)(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    
    if dropout_ratio:
        uconv1 = Dropout(dropout_ratio)(uconv1)        
    # uconv1 = Conv1D(start_neurons * 1, filter_size, activation=None, padding="same")(uconv1)
    uconv1 = se_bottleneck(uconv1,start_neurons * 1)
    
    if dropout_ratio:
        uconv1 = Dropout(dropout_ratio)(uconv1)

    output_layer = Conv1D(nClasses, 1, activation='softmax', padding='same')(uconv1)#kernel_initializer='he_normal'
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model


if __name__ == '__main__':
    model = build_model(start_neurons=32, dropout_ratio=0, filter_size=1, nClasses=3)
    model.summary()