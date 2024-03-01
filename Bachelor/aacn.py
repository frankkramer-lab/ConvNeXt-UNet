#==============================================================================#
#  Author:       Jonas Waibel                                                  #
#  Copyright:    2023 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                     Reference:                      #
#         Fabian Isensee, Klaus H. Maier-Hein.        #
#                     6 Aug 2019.                     #
#         An attempt at beating the 3D U-Net.         #
#           https://arxiv.org/abs/1908.02182          #
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
import tensorflow as tf

from tensorflow import keras
from keras.initializers import Constant
from keras.layers import Activation, LayerNormalization, Dense, multiply, add, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, DepthwiseConv2D, UpSampling2D
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D
from keras.layers import Input, Concatenate
from keras.models import Model

from tensorflow_addons.layers import StochasticDepth


class Architecture():
    # ---------------------------------------------#
    #                Initialization               #
    # ---------------------------------------------#
    def __init__(self, conv_layer_activation="relu"):
        self.conv_layer_activation = conv_layer_activation
        # Create lists of filters
        self.feature_map = [64, 96, 96, 192, 384, 768]
        self.feature_map_decoding = [64, 64, 96, 96, 192]


    def create_model_2D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Stem layer
        neurons = self.feature_map[0]
        cnn_chain = conv_layer_2D(cnn_chain, neurons, activation="relu")
        contracting_convs.append(cbam_2D(cnn_chain, neurons))

        neurons = self.feature_map[1]
        cnn_chain = Conv2D(filters=neurons, kernel_size=(4, 4), padding="valid", strides=(4, 4))(cnn_chain)
        cnn_chain = LayerNormalization(epsilon=1e-6)(cnn_chain)
        contracting_convs.append(cbam_2D(cnn_chain, neurons))

        # First contracting layer with convnext
        neurons = self.feature_map[2]
        cnn_chain = convnext_block_2D(cnn_chain, neurons)
        contracting_convs.append(cbam_2D(cnn_chain, neurons))

        # Remaining contracting layers with downsampling
        for i in range(3, len(self.feature_map)):
            neurons = self.feature_map[i]
            cnn_chain = downsample_2D(cnn_chain, neurons)
            cnn_chain = convnext_block_2D(cnn_chain, neurons)
            if i < len(self.feature_map) - 1:
                contracting_convs.append(cbam_2D(cnn_chain, neurons))

        # Middle layer
        neurons = self.feature_map[-1]
        cnn_chain = cbam_2D(cnn_chain, neurons)

        # Expanding layers
        for i in reversed(range(0, len(self.feature_map)-1)):
            neurons = self.feature_map_decoding[i]
            if i == 1:
                cnn_chain = UpSampling2D(size=(1, 1), interpolation="bilinear")(cnn_chain)
            elif i == 0:
                cnn_chain = UpSampling2D(size=(4, 4), interpolation="bilinear")(cnn_chain)
            else:
                cnn_chain = UpSampling2D(size=(2, 2), interpolation="bilinear")(cnn_chain)
            cnn_chain = tf.concat([cnn_chain, contracting_convs[i]], axis=-1)
            cnn_chain = conv_layer_2D(cnn_chain, neurons, self.conv_layer_activation)
            if i == 0:
                cnn_chain = Conv2D(filters=n_labels, kernel_size=(1, 1), activation="softmax")(cnn_chain)
            else:
                cnn_chain = cbam_2D(cnn_chain, neurons)

        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[cnn_chain])
        # Return model
        return model


    def create_model_3D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Stem layer
        neurons = self.feature_map[0]
        cnn_chain = conv_layer_3D(cnn_chain, neurons, self.conv_layer_activation)
        contracting_convs.append(cbam_3D(cnn_chain, neurons))

        neurons = self.feature_map[1]
        cnn_chain = Conv3D(filters=neurons, kernel_size=(4, 4, 4), padding="valid", strides=(4, 4))(cnn_chain)
        cnn_chain = LayerNormalization(epsilon=1e-6)(cnn_chain)
        contracting_convs.append(cbam_3D(cnn_chain, neurons))

        # First contracting layer with convnext
        neurons = self.feature_map[2]
        cnn_chain = convnext_block_3D(cnn_chain, neurons)
        contracting_convs.append(cbam_3D(cnn_chain, neurons))

        # Remaining contracting layers with downsampling
        for i in range(3, len(self.feature_map)):
            neurons = self.feature_map[i]
            cnn_chain = downsample_3D(cnn_chain, neurons)
            cnn_chain = convnext_block_3D(cnn_chain, neurons)
            if i < len(self.feature_map) - 1:
                contracting_convs.append(cbam_3D(cnn_chain, neurons))

        # Middle layer
        neurons = self.feature_map[-1]
        cnn_chain = cbam_3D(cnn_chain, neurons)

        # Expanding layers
        for i in reversed(range(0, len(self.feature_map)-1)):
            neurons = self.feature_map_decoding[i]
            if i == 1:
                cnn_chain = UpSampling3D(size=(1, 1, 1), interpolation="bilinear")(cnn_chain)
            elif i == 0:
                cnn_chain = UpSampling3D(size=(4, 4, 4), interpolation="bilinear")(cnn_chain)
            else:
                cnn_chain = UpSampling3D(size=(2, 2, 2), interpolation="bilinear")(cnn_chain)
            cnn_chain = tf.concat([cnn_chain, contracting_convs[i]], axis=-1)
            cnn_chain = conv_layer_3D(cnn_chain, neurons, self.conv_layer_activation)
            if i == 0:
                cnn_chain = Conv3D(filters=n_labels, kernel_size=(1, 1, 1), activation="softmax")(cnn_chain)
            else:
                cnn_chain = cbam_3D(cnn_chain, neurons)

        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[cnn_chain])
        # Return model
        return model

#-----------------------------------------------------#
#                   Subroutines 2D                    #
#-----------------------------------------------------#
def conv_layer_2D(input, neurons, activation, strides=1):
    conv = Conv2D(filters=neurons, activation=activation, kernel_size=(4, 4), padding="same", strides=(1, 1))(input)
    return conv


def downsample_2D(input, neurons):
    conv = LayerNormalization(epsilon=1e-6)(input)
    conv = Conv2D(filters=neurons, kernel_size=(2, 2), padding="valid", strides=(2, 2))(conv)
    return conv


#-----------------------------------------------------#
#                   Subroutines 3D                    #
#-----------------------------------------------------#
def conv_layer_3D(input, neurons, activation, strides=1):
    conv = Conv2D(filters=neurons, kernel_size=(4, 4, 4), padding="same", strides=(1, 1, 1))(input)

    return Activation(activation)(conv)


def downsample_3D(input, neurons):
    conv = LayerNormalization(epsilon=1e-6)(input)
    conv = Conv2D(filters=neurons, kernel_size=(2, 2, 2), padding="valid", strides=(2, 2, 2))(conv)

    return conv

#-----------------------------------------------------#
#                   ConvNeXt                          #
#-----------------------------------------------------#

class LayerScale(keras.layers.Layer):
    
    def __init__(self, init_value, neurons, **kwargs):
        super().__init__(**kwargs)
        self.init_value = init_value
        self.neurons = neurons
        self.gamma = self.add_weight(
            shape=(self.neurons,),
            initializer=Constant(self.init_value),
            trainable=True,
        )

    def call(self, input):
        return input * self.gamma


def convnext_block_2D(input, neurons, drop_path_rate=0.3, layer_scale_init_value=1e-6):
    conv = Conv2D(filters=neurons, kernel_size=(7, 7), padding="same", groups=neurons)(input)
    conv = LayerNormalization(epsilon=1e-6)(conv)
    conv = Conv2D(filters=neurons*4, kernel_size=(1, 1), padding="same")(conv)
    conv = Activation("gelu")(conv)
    conv = Conv2D(filters=neurons, kernel_size=(1, 1), padding="same")(conv)


    if layer_scale_init_value > 0:
        conv = LayerScale(layer_scale_init_value, neurons)(conv)

    if drop_path_rate > 0.0:
        conv = StochasticDepth(1-drop_path_rate)([input, conv])
    else:
        conv = Activation("linear")(conv)

    return input + conv

def convnext_block_3D(input, neurons, drop_path=0.0, layer_scale_init_value=1e-6):
    conv = Conv3D(kernel_size=(7, 7, 7), padding="same", groups=neurons)(input)
    conv = LayerNormalization(epsilon=1e-6)(conv)
    conv = Conv3D(units=neurons*4, kernel_size=(1, 1, 1), padding="same")(conv)
    conv = Activation("gelu")(conv)
    conv = Conv3D(units=neurons, kernel_size=(1, 1, 1), padding="same")(conv)


    if layer_scale_init_value > 0:
        gamma = tf.Variable(layer_scale_init_value * tf.ones((neurons,)))
        conv = multiply([conv, gamma])

    if drop_path > 0.0:
        conv = StochasticDepth(1-drop_path)(conv)
    else:
        conv = Activation("linear")(conv)


    return input + conv


#---------------------------------------------#
# Convolutional Block Attention Module (CBAM) #
#---------------------------------------------#
def channel_attention_2D(input_feature, neurons, ratio=8):
    kernel_initializer = tf.keras.initializers.VarianceScaling()
    bias_initializer = tf.constant_initializer(value=0.0)
    channel = neurons
    avg_pool = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)
        
    assert avg_pool.get_shape()[1:] == (1,1,channel)
    avg_pool = Dense(units=channel//ratio,
                                 activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer)(avg_pool)
    assert avg_pool.get_shape()[1:] == (1,1,channel//ratio)
    avg_pool = Dense(units=channel,                             
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer)(avg_pool)     
    assert avg_pool.get_shape()[1:] == (1,1,channel)

    max_pool = tf.reduce_max(input_feature, axis=[1,2], keepdims=True)    
    assert max_pool.get_shape()[1:] == (1,1,channel)
    max_pool = Dense(units=channel//ratio,
                                 activation=tf.nn.relu)(max_pool)  
    assert max_pool.get_shape()[1:] == (1,1,channel//ratio)
    max_pool = Dense(units=channel)(max_pool)  
    assert max_pool.get_shape()[1:] == (1,1,channel)

    scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
    
    return input_feature * scale


def channel_attention_3D(input, neurons, ratio=8):
    # Average pooling
    avg_pool = tf.reduce_mean(input, axis=[1, 2, 3], keepdims=True)

    avg_pool = Dense(units=neurons/ratio, activation="relu")(avg_pool)
    avg_pool = Dense(units=neurons, activation="relu")(avg_pool)

    # Max pooling
    max_pool = tf.reduce_max(input, axis=[1, 2, 3], keepdims=True)

    max_pool = Dense(units=neurons/ratio, activation="relu")(max_pool)
    max_pool = Dense(units=neurons, activation="relu")(max_pool)

    # Sigmoid
    sigmoid = tf.sigmoid(avg_pool + max_pool)

    re_weight = input * sigmoid

    return re_weight

def spatial_attention_2D(input_feature):
    kernel_size = 7
    kernel_initializer = tf.keras.initializers.VarianceScaling()
    avg_pool = tf.reduce_mean(input_feature, axis=[3], keepdims=True)
    assert avg_pool.get_shape()[-1] == 1
    max_pool = tf.reduce_max(input_feature, axis=[3], keepdims=True)
    assert max_pool.get_shape()[-1] == 1
    concat = tf.concat([avg_pool,max_pool], 3)
    assert concat.get_shape()[-1] == 2
    
    concat = Conv2D(filters=1,
                              kernel_size=[kernel_size,kernel_size],
                              strides=[1,1],
                              padding="same",
                              activation=None,
                              kernel_initializer=kernel_initializer,
                              use_bias=False)(concat)
    assert concat.get_shape()[-1] == 1
    concat = tf.sigmoid(concat, 'sigmoid')
    
    return input_feature * concat


def spatial_attention_3D(input):
    # Average pooling
    avg_pool = tf.reduce_mean(input, axis=[4], keepdims=True)

    # Max pooling
    max_pool = tf.reduce_max(input, axis=[4], keepdims=True)

    # Concatenation and 7x7 Convolution
    concat = tf.concat([avg_pool, max_pool], axis=4)
    conv = Conv2D(filters=1, kernel_size=(7, 7, 7), padding="same", strides=(1, 1, 1))(concat)

    # Sigmoid
    sigmoid = tf.sigmoid(conv)

    re_weight = input * sigmoid

    return re_weight


def cbam_2D(input, neurons, ratio=8):
    # Perform channel and spatial attention
    cbam = channel_attention_2D(input, neurons, ratio)
    cbam = spatial_attention_2D(cbam)

    return cbam

def cbam_3D(input, neurons, ratio=8):
    # Perform channel and spatial attention
    cbam = channel_attention_3D(input, neurons, ratio)
    cbam = spatial_attention_3D(cbam, neurons)

    return cbam














