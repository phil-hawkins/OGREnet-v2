import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
)

def edge_update(edge__in_size, node_size, u_size, edge_out_size=10, name="edge_update"):
    feature_size = (2 * node_size) + u_size
    edge_in = Input(
        shape=(feature_size, ), 
        dtype='float32',
        name=name+"_input") 

    x = Dense(1024, activation='elu')(edge_in)
    x = Dense(1024, activation='elu')(x)
    x = Dense(1024, activation='elu')(x)
    x = Dense(1024, activation='elu')(x)
    x = Dense(edge_out_size, activation='linear')(x)

    return tf.keras.Model(inputs=[edge_in], outputs=[x], name=name)

def node_update(name="edge_update", node_in_size, edge_size, u_size, node_out_size, name="node_update"):
    feature_size = (2 * node_size) + u_size
    edge_in = Input(
        shape=(none, feature_size), 
        dtype='float32',
        name=name+"_input") 


