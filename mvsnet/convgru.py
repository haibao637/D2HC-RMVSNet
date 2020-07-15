#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Convolutional GRU module.
"""

import tensorflow as tf
# from tensorflow.contrib.rnn import  nn_ops, math_ops, array_ops


def group_norm(input_tensor,
               name,
               channel_wise=True,
               group=32,
               group_channel=8,
               is_reuse=False):

    x = tf.transpose(input_tensor, [0, 3, 1, 2])

    # shapes and groups
    shape = tf.shape(x)
    N = shape[0]
    C = x.get_shape()[1]
    H = shape[2]
    W = shape[3]
    if channel_wise:
        G = max(1, C / group_channel)
    else:
        G = min(group, C)

    # use layer normalization to simplify operations
    if G == 1:
        return tf.contrib.layers.layer_norm(input_tensor)

    # use instance normalization to simplify operations
    elif G >= C:
        return tf.contrib.layers.instance_norm(input_tensor)
    
    # group normalization as suggested in the paper
    else:
        x = tf.reshape(x, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + 1e-5)

        # per channel scale and bias (gamma and beta)
        with tf.variable_scope(name + '/gn', reuse=is_reuse):        
            gamma = tf.get_variable('gamma', [C], dtype=tf.float32, initializer=tf.ones_initializer())
            beta = tf.get_variable('beta', [C], dtype=tf.float32, initializer=tf.zeros_initializer())
            
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        output = tf.transpose(output, [0, 2, 3, 1])
        return output
def convs( inputs, filters, kernel, padding='same', name='conv',reuse=tf.AUTO_REUSE):
    conv0 = tf.layers.conv2d(
                    inputs, filters/2, 5,dilation_rate=1, padding='same', name=name+"_dilate",reuse=tf.AUTO_REUSE)
    conv1 = tf.layers.conv2d(
                    inputs,filters/2, 5,dilation_rate=2, padding='same', name=name+"_dilate",reuse=tf.AUTO_REUSE)
    conv2 = tf.layers.conv2d(
                    inputs,filters/2, 5,dilation_rate=4, padding='same', name=name+"_dilate",reuse=tf.AUTO_REUSE)
    conv3 = tf.layers.conv2d(
                    inputs,filters/2, 5,dilation_rate=8, padding='same', name=name+"_dilate",reuse=tf.AUTO_REUSE)
    conv=tf.concat([conv0,conv1,conv2,conv3],-1)
    conv=tf.nn.relu(conv)
    conv=group_norm(conv, name=name+'_group_norm', group_channel=16,is_reuse=tf.AUTO_REUSE)
    conv = tf.layers.conv2d(
                    conv,filters, kernel,dilation_rate=1, padding='same', name='conv',reuse=tf.AUTO_REUSE)
    return conv


class ConvGRUCell(tf.contrib.rnn.RNNCell):
    """A GRU cell with convolutions instead of multiplications."""

    def __init__(self,
                 shape,
                 filters,
                 kernel,
                 initializer=None,
                 activation=tf.tanh,
                 normalize=True,
                 name=None,
                 scope=None
                 ):
        self._filters = filters
        self._kernel = kernel
        self._initializer = initializer
        self._activation = activation
        self._size = tf.TensorShape(shape + [self._filters])
        self._normalize = normalize
        self._feature_axis = self._size.ndims
        self._name=name
        self._scope=scope
    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def __call__(self, x, h, scope=None):
        # scope= scope if scope is not None else self._scope
        if scope is None:
            scope=self._scope
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            with tf.variable_scope('Gates'):
                
                # concatenation
                inputs = tf.concat([x, h], axis=self._feature_axis)

                # convolution
                conv = tf.layers.conv2d(
                    inputs, 2 * self._filters, self._kernel, padding='same', name='conv',reuse=tf.AUTO_REUSE)
                reset_gate, update_gate = tf.split(conv, 2, axis=self._feature_axis)

                # group normalization, actually is 'instance normalization' as to save GPU memory 
                reset_gate = group_norm(reset_gate, 'reset_norm', group_channel=16,is_reuse=tf.AUTO_REUSE)
                update_gate = group_norm(update_gate, 'update_norm', group_channel=16,is_reuse=tf.AUTO_REUSE)

                # activation
                reset_gate = tf.sigmoid(reset_gate)
                update_gate = tf.sigmoid(update_gate)

            with tf.variable_scope('Output'):

                # concatenation
                inputs = tf.concat([x, reset_gate * h], axis=self._feature_axis)

                # convolution
                conv = tf.layers.conv2d(
                    inputs, self._filters, self._kernel, padding='same', name='output_conv',reuse=tf.AUTO_REUSE)

                # group normalization
                conv = group_norm(conv, 'output_norm', group_channel=16,is_reuse=tf.AUTO_REUSE)
                    
                # activation
                y = self._activation(conv)

                # soft update
                output = update_gate * h + (1 - update_gate) * y

            return output, output

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
class BGRUCell(tf.contrib.rnn.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  Args:
    num_units: int, The number of units in the GRU cell.
    activation: Nonlinearity to use.  Default: `tanh`.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    kernel_initializer: (optional) The initializer to use for the weight and
    projection matrices.
    bias_initializer: (optional) The initializer to use for the bias.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
  """

  def __init__(self,
               num_units,
               activation=None,
               shape=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None,
               name=None):
    super(BGRUCell, self).__init__(_reuse=reuse, name=name)

    # Inputs must be 2-dimensional.

    self._name=name
    self._num_units = num_units
    self._activation = activation or tf.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._num_units=num_units
    self._size=tf.TensorShape(shape + [self._num_units])
  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def build(self, inputs_shape):


    # self._gate_kernel = self.add_variable(
    #     "gates/%s" % _WEIGHTS_VARIABLE_NAME,
    #     shape=[input_depth + 2*self._num_units, 2 * self._num_units],
    #     initializer=self._kernel_initializer)
    # self._gate_bias = self.add_variable(
    #     "gates/%s" % _BIAS_VARIABLE_NAME,
    #     shape=[ 2*self._num_units],
    #     initializer=(
    #         self._bias_initializer
    #         if self._bias_initializer is not None
    #         else tf.constant_initializer(1.0, dtype=self.dtype)))
    # self._candidate_kernel = self.add_variable(
    #     "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
    #     shape=[input_depth + 2*self._num_units, self._num_units],
    #     initializer=self._kernel_initializer)
    # self._candidate_bias = self.add_variable(
    #     "candidate/%s" % _BIAS_VARIABLE_NAME,
    #     shape=[self._num_units],
    #     initializer=(
    #         self._bias_initializer
    #         if self._bias_initializer is not None
    #         else tf.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    # inputs=#batch_size,time_steps,d,channels
    with tf.name_scope(self._name) as scope,tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        gate_inputs=tf.layers.conv1d(tf.concat([inputs,state],-1),self._num_units*2,3,padding='SAME',reuse=tf.AUTO_REUSE,name='gate_conv')
        # gate_inputs = tf.contrib.layers.group_norm(gate_inputs, self._num_units, axis=-1)
    # gate_inputs = tf.matmul(
    #     tf.concat([inputs, state], 1), self._gate_kernel)
    # gate_inputs = tf.add(gate_inputs, self._gate_bias)

        value = tf.sigmoid(gate_inputs)
        r, u = tf.split(value=value, num_or_size_splits=2, axis=-1)

        r_state = r * state
        candidate=tf.layers.conv1d(tf.concat([inputs, r_state], -1),self._num_units,3,padding='SAME',reuse=tf.AUTO_REUSE,name='candiate_conv')
        # candidate=tf.contrib.layers.group_norm(candidate,self._num_units,axis=-1)
        # candidate = tf.matmul(
        #     tf.concat([inputs, r_state], -1), self._candidate_kernel)
        # candidate = tf.add(candidate, self._candidate_bias)

        c = self._activation(candidate)
        new_h = u * state + (1 - u) * c
    return new_h, new_h


