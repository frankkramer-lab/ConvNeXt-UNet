#==============================================================================#
#  Author:       Jonas Waibel                                                  #
#  Copyright:    2024 IT-Infrastructure for Translational Medical Research,    #
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

# External libraries
import tensorflow as tf

from keras.models import Model
from keras.layers import Input, concatenate, Dense
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import BatchNormalization
# Internal libraries/scripts
from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture

#-----------------------------------------------------#
#         Architecture class: CBAM U-Net              #
#-----------------------------------------------------#
""" CBAM U-Net architecture.

Methods:
    __init__                Object creation function
    create_model_2D:        Creating the 2D CBAM U-Net model using Keras
    create_model_3D:        Creating the 3D CBAM U-Net model using Keras
"""
class Architecture(Abstract_Architecture):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, n_filters=32, depth=4, activation='softmax',
                 batch_normalization=True):
        # Parse parameter
        self.n_filters = n_filters
        self.depth = depth
        self.activation = activation
        # Batch normalization settings
        self.ba_norm = batch_normalization
        self.ba_norm_momentum = 0.99

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Contracting Layers
        for i in range(0, self.depth):
            neurons = self.n_filters * 2**i
            cnn_chain, last_conv = contracting_layer_2D(cnn_chain, neurons,
                                                        self.ba_norm,
                                                        self.ba_norm_momentum)
            contracting_convs.append(cbam_2D(last_conv, neurons))

        # Middle Layer
        neurons = self.n_filters * 2**self.depth
        cnn_chain = middle_layer_2D(cnn_chain, neurons, self.ba_norm,
                                    self.ba_norm_momentum)

        # Expanding Layers
        for i in reversed(range(0, self.depth)):
            neurons = self.n_filters * 2**i
            cnn_chain = expanding_layer_2D(cnn_chain, neurons,
                                           contracting_convs[i], self.ba_norm,
                                           self.ba_norm_momentum)

        # Output Layer
        conv_out = Conv2D(n_labels, (1, 1),
                   activation=self.activation)(cnn_chain)
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model

    #---------------------------------------------#
    #               Create 3D Model               #
    #---------------------------------------------#
    def create_model_3D(self, input_shape, n_labels=2):
        # Input layer
        inputs = Input(input_shape)
        # Start the CNN Model chain with adding the inputs as first tensor
        cnn_chain = inputs
        # Cache contracting normalized conv layers
        # for later copy & concatenate links
        contracting_convs = []

        # Contracting Layers
        for i in range(0, self.depth):
            neurons = self.n_filters * 2**i
            cnn_chain, last_conv = contracting_layer_3D(cnn_chain, neurons,
                                                        self.ba_norm,
                                                        self.ba_norm_momentum)
            contracting_convs.append(last_conv)

        # Middle Layer
        neurons = self.n_filters * 2**self.depth
        cnn_chain = middle_layer_3D(cnn_chain, neurons, self.ba_norm,
                                    self.ba_norm_momentum)

        # Expanding Layers
        for i in reversed(range(0, self.depth)):
            neurons = self.n_filters * 2**i
            cnn_chain = expanding_layer_3D(cnn_chain, neurons,
                                           contracting_convs[i], self.ba_norm,
                                           self.ba_norm_momentum)

        # Output Layer
        conv_out = Conv3D(n_labels, (1, 1, 1),
                   activation=self.activation)(cnn_chain)
        # Create Model with associated input and output layers
        model = Model(inputs=[inputs], outputs=[conv_out])
        # Return model
        return model

#-----------------------------------------------------#
#                   Subroutines 2D                    #
#-----------------------------------------------------#
# Create a contracting layer
def contracting_layer_2D(input, neurons, ba_norm, ba_norm_momentum):
    conv1 = Conv2D(neurons, (3,3), activation='relu', padding='same')(input)
    if ba_norm : conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conv2 = Conv2D(neurons, (3,3), activation='relu', padding='same')(conv1)
    if ba_norm : conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    return pool, conv2

# Create the middle layer between the contracting and expanding layers
def middle_layer_2D(input, neurons, ba_norm, ba_norm_momentum):
    conv_m1 = Conv2D(neurons, (3, 3), activation='relu', padding='same')(input)
    if ba_norm : conv_m1 = BatchNormalization(momentum=ba_norm_momentum)(conv_m1)
    conv_m2 = Conv2D(neurons, (3, 3), activation='relu', padding='same')(conv_m1)
    if ba_norm : conv_m2 = BatchNormalization(momentum=ba_norm_momentum)(conv_m2)
    return conv_m2

# Create an expanding layer
def expanding_layer_2D(input, neurons, concatenate_link, ba_norm,
                       ba_norm_momentum):
    up = concatenate([Conv2DTranspose(neurons, (2, 2), strides=(2, 2),
                     padding='same')(input), concatenate_link], axis=-1)
    conv1 = Conv2D(neurons, (3, 3,), activation='relu', padding='same')(up)
    if ba_norm : conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conv2 = Conv2D(neurons, (3, 3), activation='relu', padding='same')(conv1)
    if ba_norm : conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    return conv2

#-----------------------------------------------------#
#                   Subroutines 3D                    #
#-----------------------------------------------------#
# Create a contracting layer
def contracting_layer_3D(input, neurons, ba_norm, ba_norm_momentum):
    conv1 = Conv3D(neurons, (3,3,3), activation='relu', padding='same')(input)
    if ba_norm : conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conv2 = Conv3D(neurons, (3,3,3), activation='relu', padding='same')(conv1)
    if ba_norm : conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    pool = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    return pool, conv2

# Create the middle layer between the contracting and expanding layers
def middle_layer_3D(input, neurons, ba_norm, ba_norm_momentum):
    conv_m1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(input)
    if ba_norm : conv_m1 = BatchNormalization(momentum=ba_norm_momentum)(conv_m1)
    conv_m2 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(conv_m1)
    if ba_norm : conv_m2 = BatchNormalization(momentum=ba_norm_momentum)(conv_m2)
    return conv_m2

# Create an expanding layer
def expanding_layer_3D(input, neurons, concatenate_link, ba_norm,
                       ba_norm_momentum):
    up = concatenate([Conv3DTranspose(neurons, (2, 2, 2), strides=(2, 2, 2),
                     padding='same')(input), concatenate_link], axis=4)
    conv1 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(up)
    if ba_norm : conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conv2 = Conv3D(neurons, (3, 3, 3), activation='relu', padding='same')(conv1)
    if ba_norm : conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    return conv2


#---------------------------------------------#
# Convolutional Block Attention Module (CBAM) #
#---------------------------------------------#


def channel_attention_2D(input, neurons, ratio=8):
    # Average pooling
    avg_pool = GlobalAveragePooling2D(keepdims=True)(input)

    # Max pooling
    max_pool = GlobalMaxPooling2D(keepdims=True)(input)

    # MLP
    avg_pool = Dense(units=neurons // ratio, activation="relu")(avg_pool)
    avg_pool = Dense(units=neurons)(avg_pool)

    max_pool = Dense(units=neurons // ratio, activation="relu")(max_pool)
    max_pool = Dense(units=neurons)(max_pool)

    # Sigmoid
    sigmoid = tf.sigmoid(avg_pool + max_pool)

    re_weight = input * sigmoid

    return re_weight


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


def spatial_attention_2D(input):
    # Average pooling
    avg_pool = tf.reduce_mean(input, axis=[3], keepdims=True)

    # Max pooling
    max_pool = tf.reduce_max(input, axis=[3], keepdims=True)

    # Concatenation and 7x7 Convolution
    concat = tf.concat([avg_pool, max_pool], axis=3)
    conv = Conv2D(filters=1, kernel_size=(7, 7), padding="same", strides=(1, 1))(concat)

    # Sigmoid
    sigmoid = tf.sigmoid(conv)

    re_weight = input * sigmoid

    return re_weight


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
    cbam = spatial_attention_3D(cbam)

    return cbam
