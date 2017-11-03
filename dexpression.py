import numpy as np
import keras
from keras.models import Sequential
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dense, Dropout, Reshape, Permute, merge, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from imagenet_utils import _obtain_input_shape
from keras.layers.normalization import BatchNormalization

from __future__ import print_function
from __future__ import absolute_import

def DeXpression(include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                padding=None,
                classes=7):
    # Check weights
    if weights not in {'dexpression', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `dexpression` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 7:
        raise ValueError('If using `weights` as dexpression with `include_top`'
                         ' as true, `classes` should be 7')
    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=139,
        data_format=K.image_data_format(),
        include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    # START MODEL
    conv_1 = Convolution2D(64, (7, 7), strides=(2, 2), padding=padding, activation='relu', name='conv_1')(img_input)
    maxpool_1 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    x = BatchNormalization()(maxpool_1)
    
    # FEAT-EX1
    conv_2a = Convolution2D(96, (1, 1), strides=(1,1), activation='relu', padding=padding, name='conv_2a')(x)
    conv_2b = Convolution2D(208, (3, 3), strides=(1,1), activation='relu', padding=padding, name='conv_2b')(conv_2a)
    maxpool_2a = MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_2a')(x)
    conv_2c = Convolution2D(64, (1, 1), strides=(1,1), name='conv_2c')(maxpool_2a)
    concat_1 = merge([conv_2b,conv_2c],mode='concat',concat_axis=3,name='concat_2')
    maxpool_2b = MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_2b')(concat_1)
    
    # FEAT-EX2
    conv_3a = Convolution2D(96, (1, 1), strides=(1,1), activation='relu', padding=padding, name='conv_3a')(maxpool_2b)
    conv_3b = Convolution2D(208, (3, 3), strides=(1,1), activation='relu', padding=padding, name='conv_3b')(conv_3a)
    maxpool_3a = MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_3a')(maxpool_2b)
    conv_3c = Convolution2D(64, (1, 1), strides=(1,1), name='conv_3c')(maxpool_2a)
    concat_3 = merge([conv_3b,conv_3c],mode='concat',concat_axis=3,name='concat_3')
    maxpool_3b = MaxPooling2D((3,3), strides=(1,1), padding=padding, name='maxpool_3b')(concat_3)
    
    # FINAL LAYERS
    net = Flatten()(maxpool_3b)
    net = Dense(classes, activation='softmax', name='predictions')(net)
    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, net, name='deXpression')
    return model
