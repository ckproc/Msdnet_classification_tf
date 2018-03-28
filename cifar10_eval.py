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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import os
import sys
import argparse
import numpy as np
import tensorflow as tf

import msdnet

FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ckp/Msd_tf/weights',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

'''
def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

      #summary = tf.Summary()
      #summary.ParseFromString(sess.run(summary_op))
      #summary.value.add(tag='Precision @ 1', simple_value=precision)
      #summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
'''
def computeScore(prediction,labels_test):
    #32*10
    top1_count = 0
    batchsize = prediction.shape[0]
    for i in range(batchsize):
      sort_indices = np.argsort(prediction[i])
      if labels_test[i] == sort_indices[-1]:
        top1_count += 1
    return top1_count

def main(args):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
   with tf.device('/cpu:1'):
    with tf.variable_scope("model") as scope:
        # Get images and labels for CIFAR-10.
        #eval_data = FLAGS.eval_data == 'test'
        images, labels = msdnet.inputs(args,True)
        images = tf.identity(images, 'input')
        print (images)
        # Build a Graph that computes the logits predictions from the
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits,_ = msdnet.inference(images,args)

        # Calculate predictions.
        #top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            msdnet.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
              # Restores from checkpoint
              saver.restore(sess, ckpt.model_checkpoint_path)
              # Assuming model_checkpoint_path looks something like:
              #   /my-favorite-path/cifar10_train/model.ckpt-0,
              # extract global_step from it.
              #global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
              print('No checkpoint file found')
              return
            
            num_examples=10000
            num_iter = int(math.ceil(num_examples / args.batch_size))
            test_step=0
            N = 0.0
            total_sample_count = num_iter * args.batch_size
            total=0.0
            while test_step<num_iter:
              predictions,targets = sess.run([logits,labels])
              for i in range(len(predictions)):
                top1 = computeScore(predictions[i],targets)
                total+=top1
              N=N+args.batch_size
              test_step+=1
        
        print(total,'/',N)        


#def main(argv=None):  # pylint: disable=unused-argument
  #cifar10.maybe_download_and_extract()
  #if tf.gfile.Exists(FLAGS.eval_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  #tf.gfile.MakeDirs(FLAGS.eval_dir)
  #evaluate()
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    ### ------------ General options --------------------
    parser.add_argument('--data', type=str, 
        help='Path to dataset.', default='')
    parser.add_argument('--dataset', type=str,
        help='Options: imagenet | cifar10 | cifar100.', default='imagenet')
    parser.add_argument('--manualSeed', type=int,
        help='Manually set RNG seed.', default=0)
    parser.add_argument('--gen', type=str,
        help='path to save generated files.', default='gen')
    parser.add_argument('--precision', type=str,
        help='Options: single | double | half.', default='single')
    #log_device_placement
    parser.add_argument('--log_device_placement',
        help='whether to log device placement.', action='store_true')
    parser.add_argument('--train_dir',type=str,help='directory to write checkpoints',default='/home/ckp/Msd_tf/checkpoints')
    parser.add_argument('--logs',type=str,help='directory to write summary',default='/home/ckp/Msd_tf/logs')
    parser.add_argument('--max_steps',type=int,help='',default=220000)
    parser.add_argument('--num_gpus',type=int,help='',default=1)
    parser.add_argument('--batch_size',type=int,help='',default=64)
    parser.add_argument('--nScales',type=int,help='',default=3)
    
    ###------------- Data options ------------------------
    
    #parser.add_argument('--nThreads', type=int,
    #    help='number of data loading threads.', default=2)
    parser.add_argument('--DataAug', 
        help='use data augmentation or not.', action='store_true')
        
        
    ###------------- Training options --------------------
    parser.add_argument('--testOnly', 
        help='Run on validation set only.', action='store_true')
    parser.add_argument('--tenCrop', 
        help='Ten-crop testing.', action='store_true')
    parser.add_argument('--reduction',type=float,help='dimension reduction ratio at transition layers',default=0.5)    
    ###------------- Checkpointing options ---------------
    ###---------- Optimization options ----------------------
    parser.add_argument('--LR', type=float,help='initial learning rate.', default=0.1)
    parser.add_argument('--momentum', type=float, help='.', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='.', default=1e-4)
    
    ###---------- Model options ----------------------------------
    
    #parser.add_argument('--shareGradInput',help='Share gradInput tensors to reduce memory usage',action='store_true' )
    #parser.add_argument('--optnet',help='Use optnet to reduce memory usage', action='store_true')
    
    ###---------- MSDNet MOdel options ----------------------------------
    parser.add_argument('--base', type=int,
        help='the layer to attach the first classifier', default=4)
    parser.add_argument('--nBlocks', type=int,
        help='number of blocks/classifiers', default=10)
    parser.add_argument('--stepmode', type=str,
        help='patten of span between two adjacent classifers |even|lin_grow|', default='even')
    parser.add_argument('--step', type=int,
        help='span between two adjacent classifers.', default=2)
    parser.add_argument('--bottleneck',
        help='use 1x1 conv layer or not', action='store_true')
    parser.add_argument('--growthRate', type=int,
        help='number of output channels for each layer (the first scale).', default=6)
    parser.add_argument('--grFactor', type=str,
        help='growth rate factor of each sacle', default='1-2-4-4')
    parser.add_argument('--prune', type=str,
        help='specify how to prune the network, min | max', default='max')
    parser.add_argument('--joinType', type=str,
        help='add or concat for features from different paths', default='concat')
    
    parser.add_argument('--bnFactor', type=str,
        help='bottleneck factor of each sacle, 4-4-4-4 | 1-2-4-4', default='1-2-4-4')
        
    parser.add_argument('--initChannels',type=int,help='number of features produced by the initial conv layer',default=32)
    #parser.add_argument('--EEensemble', help='use ensemble or not in early exit', action='store_true')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.msdnet')
        #tf.app.flags.DEFINE_string('data_dir', '/home/ckp/Msd_tf/dataset/cifar10_data',"""Path to the CIFAR-10 data directory.""")
    parser.add_argument('--data_dir',type=str ,default = '/home/ckp/Msd_tf/dataset/cifar10_data')
    parser.add_argument('--eval_dir',type=str ,default = '/home/ckp/Msd_tf/dataset/cifar10_data')
    return parser.parse_args(argv)

if __name__ == '__main__':
  main(parse_arguments(sys.argv[1:]))
