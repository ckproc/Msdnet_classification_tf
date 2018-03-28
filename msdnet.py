# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import math
from six.moves import urllib
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import cifar10_input_data
import cifar10_input_data

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 2,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/home/ckp/Msd_tf/dataset/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_string('eval_dir', '/home/ckp/Msd_tf/dataset/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")                           
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
 
# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input_data.IMAGE_SIZE
NUM_CLASSES = cifar10_input_data.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input_data.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input_data.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

'''
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var
'''

def distorted_inputs(args,batch_size_placeholder):
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  #print (FLAGS.data_dir)
  if not args.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(args.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input_data.distorted_inputs(args.batch_size,data_dir=data_dir,
                                                  batch_size=batch_size_placeholder)
  #if FLAGS.use_fp16:
  #  images = tf.cast(images, tf.float16)
  #  labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(args, eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not args.eval_dir:
    raise ValueError('Please supply a data_dir')
  #data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  eval_dir = os.path.join(args.eval_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input_data.inputs(args.batch_size,eval_data=eval_data,
                                        data_dir=eval_dir,
                                        batch_size=args.batch_size)
  #if FLAGS.use_fp16:
  #  images = tf.cast(images, tf.float16)
  #  labels = tf.cast(labels, tf.float16)
  return images, labels

'''
def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=None)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear
'''
  
def build(input, nChannels, nOutChannels, type, bottleneck, bnWidth):
    innerChannels = nChannels
    if not bnWidth :
      bnWidth = 4
    with tf.variable_scope(type):
        if bottleneck:
          innerChannels = min(innerChannels, bnWidth * nOutChannels)
          input = slim.conv2d(input, innerChannels, 1, stride=1, padding='VALID',normalizer_fn =slim.batch_norm, scope='conv_bottleneck');
        if type == 'normal':
          output = slim.conv2d(input, nOutChannels, 3,stride=1,padding='SAME',normalizer_fn =slim.batch_norm,scope = 'conv_normal')
        elif type == 'down':
          output = slim.conv2d(input, nOutChannels, 3,stride=2,padding='SAME',normalizer_fn =slim.batch_norm,scope = 'conv_dowm')
        elif type == 'up':
          output = slim.conv2d_transpose(input, nOutChannels, 3,stride=2,padding='SAME',normalizer_fn =slim.batch_norm,scope = 'conv_up')
    
    return output
    
    
    
def build_net_normal(input, nChannels, nOutChannels, bottleneck, bnWidth):
    #print (input)
    output1 = build(input[0], nChannels, nOutChannels, 'normal', bottleneck, bnWidth)
    output = tf.concat([input[0], output1], 3)
    return output
    
  
def build_net_down_normal(input, nChannels1, nChannels2, nOutChannels, bottleneck, bnWidth1, bnWidth2, scale,isTrans):
    assert nOutChannels % 2 == 0, 'Growth rate invalid!'
    if isTrans:
      output1 = build(input[scale], nChannels1, nOutChannels/2, 'down', bottleneck, bnWidth1)
      output2 = build(input[scale+1], nChannels2, nOutChannels/2, 'normal', bottleneck, bnWidth2)
      output = tf.concat([input[scale+1], output1, output2], 3)
    else:
      output1 = build(input[scale-1], nChannels1, nOutChannels/2, 'down', bottleneck, bnWidth1)
      output2 = build(input[scale], nChannels2, nOutChannels/2, 'normal', bottleneck, bnWidth2)
      output = tf.concat([input[scale], output1, output2], 3)
    return output
    
def MSDNet_Layer_first(input, Cins, Couts, args):
    '''
     input : a tensor (orginal image)
     output: a table of nScale tensors
     
    '''
    output=[]
    #print (args.nScales)
    
    for i in range(args.nScales):
        with tf.variable_scope("scale%d" %i):
          if(i==0):
            output_s = slim.conv2d(input, Couts*args.grFactor[0], 3, stride=1, padding='SAME', normalizer_fn =slim.batch_norm, scope='conv1')
            
          else:
            output_s = slim.conv2d(output_s, Couts*args.grFactor[i], 3, stride=2, padding='SAME', normalizer_fn =slim.batch_norm, scope='conv1')
          output.append(output_s)
    return output
    
def MSDNet_Layer(input, nIn, nOutc, args, inScales, outScales):
    '''
     input: a table of `nScales` tensors
     output: a table of `nScales` tensors
    '''
    #print (input)
    outputs=[]
    discard = inScales - outScales
    assert discard<=1, 'Double check inScales {0} and outScales {1}'.format(inScales,outScales)
    offset = args.nScales - outScales
    isTrans = outScales<inScales  
    print ("tran:",isTrans,'outscale:',outScales)
    for i in range(outScales):
        with tf.variable_scope("scale%d" %i):
            if i==0:
                if isTrans:
                    nIn1, nIn2, nOut = nIn*args.grFactor[offset-1], nIn*args.grFactor[offset+1-1], args.grFactor[offset+1-1]*nOutc
                    output = build_net_down_normal(input, nIn1, nIn2, nOut, args.bottleneck, args.bnFactor[offset-1], args.bnFactor[offset+1-1],i,isTrans)
                    
                else :
                    output = build_net_normal(input, nIn*args.grFactor[offset+1-1], args.grFactor[offset+1-1]*nOutc, args.bottleneck, args.bnFactor[offset+1-1])
                outputs.append(output)
            else :
                nIn1, nIn2, nOut = nIn*args.grFactor[offset+i-1], nIn*args.grFactor[offset+i], args.grFactor[offset+i]*nOutc
                output = build_net_down_normal(input,nIn1, nIn2, nOut, args.bottleneck, args.bnFactor[offset+i-1], args.bnFactor[offset+i],i,isTrans)
                outputs.append(output)
    
    #print (outputs)
    print ('------')
    return outputs
    

def build_transition(input, nIn, nOut, outScales, offset, args):
    output=[]
    with tf.variable_scope('transition'):
      for i in range(outScales):
        with tf.variable_scope('scale%d' %i):
          output_s = slim.conv2d(input[i], nOut * args.grFactor[offset + i], 1, stride=1, padding='VALID', scope='conv1')
          output.append(output_s)
    return output
         
def build_block(input, inChannels, args, step, layer_all, layer_curr,blockname):
    nIn = inChannels
    with tf.variable_scope(blockname):
        if layer_curr==0:
          input = MSDNet_Layer_first(input,3,inChannels,args)
          #print ("first layer feature size",input)
        
        
        for i in range(step):
          inScales, outScales = args.nScales, args.nScales
          layer_curr = layer_curr+1
          #add inscale outscale computation here
          if args.prune == 'min':
            inScales = min(args.nScales, layer_all - layer_curr + 2)
            outScales = min(args.nScales, layer_all - layer_curr + 1)
          elif args.prune=='max':
            interval = math.ceil(layer_all/args.nScales)
            inScales = args.nScales - math.floor((max(0, layer_curr -2))/interval)
            outScales = args.nScales - math.floor((layer_curr -1)/interval)  
          inScales = int(inScales)
          outScales = int(outScales)
          print('|', 'inScales ', inScales, 'outScales ', outScales , '|') 
          with tf.variable_scope('step_%d' %i):
            #print (len(input))
            input = MSDNet_Layer(input,nIn,args.growthRate,args,inScales,outScales)
            #print (blockname,"shape",input)
          nIn = nIn + args.growthRate
          if args.prune == 'max' and inScales > outScales and args.reduction > 0 :
            offset = args.nScales - outScales
            input = build_transition(input, nIn, math.floor(args.reduction*nIn), outScales, offset, args)
            nIn = math.floor(args.reduction*nIn)
            print('|', 'Transition layer inserted!', '\t\t|')
          elif args.prune == 'min' and args.reduction >0 and ((layer_curr == math.floor(layer_all/3) or layer_curr == math.floor(2*layer_all/3))):
            offset = args.nScales - outScales
            input = build_transition(input, nIn, math.floor(args.reduction*nIn), outScales, offset, args)
            nIn = math.floor(args.reduction*nIn)
            print('|', 'Transition layer inserted!', '\t\t|')
    #print (type(input))
    return input, nIn

    
def build_classifier_cifar(input, inChannels, nclass, scopename,args):
    interChannels1, interChannels2 = 128, 128
    #with slim.arg_scope([slim.conv2d,slim.fully_connected]),
    with tf.variable_scope(scopename):
        output = slim.conv2d(input, interChannels1, 3, stride=2,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.1),weights_regularizer=slim.l2_regularizer(args.weight_decay), scope='conv1')
        output = slim.conv2d(output, interChannels2, 3, stride=2,padding='SAME',weights_initializer=tf.truncated_normal_initializer(stddev=0.1),weights_regularizer=slim.l2_regularizer(args.weight_decay), scope='conv2')
        #print('after conv',output)
        output = slim.avg_pool2d(output, 2, stride=2, padding='SAME',scope='avgpool1')
        #print('after pool',output)
        output = slim.flatten(output,scope='flatten')
        #print('after flatten',output)
        logits = slim.fully_connected(output, nclass, activation_fn=None,normalizer_fn=None,scope='Bottleneck', reuse=False)
    return logits
    
    
  
def inference(images,args,phase_train_placeholder):
    istrain=True
    nChannels = args.initChannels
    nblocks = args.nBlocks
    nIn = nChannels
    layer_curr = 0
    layer_all = args.base
    steps=[0]*nblocks
    steps[0]=args.base
    output=[]
    for i in range(1,nblocks):
     if args.stepmode == 'even':
       steps[i]=args.step
     else :
       steps[i] = args.step*i+1
     layer_all= layer_all+steps[i]
    print("building network of steps: ")
    #print(steps)
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    end_points = []
    with slim.arg_scope([slim.batch_norm],is_training = phase_train_placeholder):
        with slim.arg_scope([slim.conv2d,slim.fully_connected],weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                            weights_regularizer=slim.l2_regularizer(args.weight_decay), normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
        #with slim.arg_scope([slim.conv2d],weights_initializer=tf.truncated_normal_initializer(stddev=0.1),weights_regularizer=slim.l2_regularizer(weight_decay), normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
         for i in range(nblocks):
          blockname = 'b_'+str(i)
          if layer_curr==0:
           net,nIn = build_block(images,nIn,args,steps[i],layer_all,layer_curr,blockname)
           #print (net)
          else:
           #print (net)
           net,nIn = build_block(net,nIn,args,steps[i],layer_all,layer_curr,blockname)
          end_points.append(net)
          layer_curr = layer_curr + steps[i]
          scopename='classifier_'+str(i)
          #print (len(net))
          #print (type(net))
          #print (net.shape)
          logits = build_classifier_cifar(net[-1],nIn*args.grFactor[args.nScales], 10 , scopename,args)
          end_points.append(logits)
          #logits = build_classifier_cifar(net[1],nIn*args.grFactor[args.nScales], 10 , scopename,args)
          output.append(logits)
    
    return output,end_points




def multi_loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    for i in range(len(logits)):
        with tf.variable_scope('logits_%d' %i):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits[i], name='cross_entropy_per_example')
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            tf.add_to_collection('losses', cross_entropy_mean)
            
    #return tf.add_n(tf.get_collection('losses'), name='class_loss')
    return tf.get_collection('losses')

'''
def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return cross_entropy_mean
  #return tf.add_n(tf.get_collection('losses'), name='total_loss')

'''
def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def computeLR(INITIAL_LEARNING_RATE, global_step, max_steps):
   decay = 0
   if global_step>int(max_steps*0.75):
     decay = 2
   elif global_step>int(max_steps*0.5):
     decay = 1
   return INITIAL_LEARNING_RATE*math.pow(0.1,decay)
   

def train(total_loss, global_step,args):
  """Train CIFAR-10 model.

  Create an argsimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / args.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  #lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
  boundaries =[0]*3
  boundaries[0]=int(args.max_steps*0.5)
  boundaries[1]=int(args.max_steps*0.75)
  boundaries[2]=int(args.max_steps*1.0)
  values =[0]*4
  for i in range(len(values)):
    #clr = INITIAL_LEARNING_RATE*math.pow(0.1,i)
    values[i]=INITIAL_LEARNING_RATE*math.pow(0.1,i)
      
  lr = tf.train.piecewise_constant(global_step,boundaries,values)
  tf.summary.scalar('learning_rate',lr)
  #lr = computeLR(INITIAL_LEARNING_RATE,global_step,args.max_steps)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    #opts = tf.train.GradientDescentOptimizer(lr)
    opts = tf.train.MomentumOptimizer(lr,args.momentum,use_nesterov = True)
    grads = opts.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opts.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op