import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1,1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (
            s[0], 
            None if s[1] is None else s[1]+2*self.padding[0],
            None if s[2] is None else s[2]+2*self.padding[1],
            s[3]
        )

    def call(self, x):
        (i_pad,j_pad) = self.padding
        return tf.pad(x, [[0,0], [i_pad,i_pad], [j_pad,j_pad], [0,0]], 'REFLECT')
