import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils import pytorch_kaiming_weight_factor

# INITIALIZATION 

# factor, mode, unifrom = pytorch_kamiming_factor(a = 0.0, uniform = False)
# weight_init = tf_contrib.layers.variable_scaling_initializer(factor = factor, mode = mode, uniform = uniform)
weight_init = tf.random_normal_initializer(mean = 0.0, stddev = 0.02)

weight_regularizer = None
weight_regularizer_fully = None

# LAYER

def conv(x, channels, kernel = 4, stride = 2, pad = 0, pad_style = 'zeros', use_bias = True, sn = False, scope = 'conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)
            
            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_style == 'zeros':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_style == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode = 'REFLECT')
        
        if sn:
            
