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
import numpy as np
import tensorflow as tf
import glob
import re
import cv2
import sys
import argparse
import msdnet
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('test_dir', '/home/ckp/Msd_tf/dataset/subset',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ckp/Msd_tf/weights',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

labels_dict = {'airplane':0,'automobile':1,'bird':2,'cat':3,'deer':4,'dog':5,'frog':6,'horse':7,'ship':8,'truck':9}

tensor_name = [[u'model/b_0/step_3/scale0/concat:0', u'model/b_0/step_3/scale1/concat:0', u'model/b_0/step_3/scale2/concat:0'], [u'model/classifier_0/Bottleneck/BiasAdd:0'], [u'model/b_1/step_1/scale0/concat:0', u'model/b_1/step_1/scale1/concat:0', u'model/b_1/step_1/scale2/concat:0'], [u'model/classifier_1/Bottleneck/BiasAdd:0'], [u'model/b_2/step_1/scale0/concat:0', u'model/b_2/step_1/scale1/concat:0', u'model/b_2/step_1/scale2/concat:0'], [u'model/classifier_2/Bottleneck/BiasAdd:0'], [u'model/b_3/step_1/scale0/concat:0', u'model/b_3/step_1/scale1/concat:0'], [u'model/classifier_3/Bottleneck/BiasAdd:0'], [u'model/b_4/step_1/scale0/concat:0', u'model/b_4/step_1/scale1/concat:0'], [u'model/classifier_4/Bottleneck/BiasAdd:0'], [u'model/b_5/step_1/scale0/concat:0', u'model/b_5/step_1/scale1/concat:0'], [u'model/classifier_5/Bottleneck/BiasAdd:0'], [u'model/b_6/step_1/scale0/concat:0', u'model/b_6/step_1/scale1/concat:0'], [u'model/classifier_6/Bottleneck/BiasAdd:0'], [u'model/b_7/step_1/scale0/concat:0'], [u'model/classifier_7/Bottleneck/BiasAdd:0'], [u'model/b_8/step_1/scale0/concat:0'], [u'model/classifier_8/Bottleneck/BiasAdd:0'], [u'model/b_9/step_1/scale0/concat:0'], [u'model/classifier_9/Bottleneck/BiasAdd:0']]
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
    
def load_model(model):
    pass
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    
        
        
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  


def computeScore(prediction,labels_test):
    #32*10
    top1_count = 0
    batchsize = prediction.shape[0]
    print (batchsize)
    for i in range(batchsize):
      sort_indices = np.argsort(prediction[i])
      #print (sort_indices)
      #print (labels_test[i])
      if labels_test[i] == sort_indices[-1]:
        top1_count += 1
    return top1_count

def softmax(x):
    x=x.astype(float)
    if x.ndim==1:
        S=np.sum(np.exp(x))
        return np.exp(x)/S
    elif x.ndim==2:
        result=np.zeros_like(x)
        M,N=x.shape
        for n in range(M):
            S=np.sum(np.exp(x[n,:]))
            result[n,:]=np.exp(x[n,:])/S
        return result
    else:
        print("The input array is not 1- or 2-dimensional.")
        
def main(args):
 if True:
  with tf.Graph().as_default() as g:
   #with tf.device("/cpu:0"):
    sess = tf.Session()
    with sess.as_default():
      model_exp = os.path.expanduser(FLAGS.checkpoint_dir)
      if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
      else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)  
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(sess, os.path.join(model_exp, ckpt_file))
      img_list = glob.glob(FLAGS.test_dir+'/*.jpg')
      images_expand = np.zeros((len(img_list), 32, 32, 3),dtype=float)
      i=0
      laa=[]
      image_placeholder = g.get_tensor_by_name('model/input:0')
     # batch_size_placeholder = g.get_tensor_by_name('model/batch_size')
      phase_train = g.get_tensor_by_name('model/phase_train:0')
      #logits = g.get_tensor_by_name(tensor_name[19][0])
      predictions=[]
      
      
      for i in range(10):
        logits = g.get_tensor_by_name(tensor_name[2*i+1][0])
        predictions.append(logits)
      
      
      for image_path in img_list:
          #print (image_path)
          img = cv2.imread(image_path)
          image_path = os.path.split(os.path.splitext(image_path)[0])[1]
          label = image_path.split('_')[0]
          label_id = int(label)
          print (label_id)
          laa.append(label_id)
          #print (label_id)
          #img_w = img.shape[1]
          #img_h = img.shape[0]
          image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
          image_np = prewhiten(image_np)
          #print (image_np)
          image_np_expanded = np.expand_dims(image_np, axis=0)
          #images_expand[i,:,:,:] = image_np
          #i+=1
          #print(images_expand)
          h = sess.partial_run_setup(predictions,[image_placeholder,phase_train])
          classifier1 = sess.partial_run(h,predictions[0],feed_dict={image_placeholder:image_np_expanded,phase_train:False})
      
          classifier1 = softmax(classifier1)
          #if np.max(classifier1)>0.95:
          print ("exit 1:",classifier1)
             #continue
          #else:
          for i in range(1,10):
             res_logits = sess.partial_run(h,predictions[i])
             #if np.max(softmax(res_logits))>0.99:
               #break
             print ("exit",i+1,":",softmax(res_logits))
      
      #classifier1 = sess.partial_run(h,predictions[0],feed_dict={image_placeholder:images_expand,phase_train:False})
      #print (classifier1)
      #classifier1 = softmax(classifier1)
      #if np.max(classifier1)>0.9:
      # print ("1:",classifier1,":",laa)
       
      #else:
      #  for i in range(1,10):
      #    res_logits = sess.partial_run(h,predictions[i])
      #    if np.max(softmax(res_logits))>0.9:
      #       break
      #    print (i,":",res_logits,":",laa)
           
         
      #res_logits = sess.run(logits,feed_dict={image_placeholder:images_expand,phase_train_placeholder:False})  
      #labels = np.array(laa)
      #top1 = computeScore(res_logits,labels)
      #print (top1)
      #print (np.argsort(res_logits[0]))
          
 else:
  with tf.Graph().as_default() as g:
   with tf.device('/cpu:0'):
    images_test,labels_test=msdnet.inputs(args, True)
    # Get images and labels for CIFAR-10.
    #eval_data = FLAGS.eval_data == 'test'
    #images, labels = cifar10.inputs(eval_data=eval_data)
    #gpu_options = tf.GPUOptions()
    #configs = tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)
   
    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)
    with sess.as_default():
      model_exp = os.path.expanduser(FLAGS.checkpoint_dir)
      if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
      else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)  
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(sess, os.path.join(model_exp, ckpt_file))
      #print ('start')
      
      #print (images_test)
      
      num_examples=10000
      num_iter = int(math.ceil(num_examples / args.batch_size))
      test_step=0
      N = 0.0
      total_sample_count = num_iter * args.batch_size
      total=0.0
      logits = g.get_tensor_by_name(tensor_name[19][0])
      image_placeholder = g.get_tensor_by_name('model/input:0')
      print ('start')
      #while test_step<num_iter:
      images,labels = sess.run([images_test,labels_test])
      print (images)
        #print (images.shape)
      for i in range(64):
        cv2.imwrite('./tmp/'+str(labels[i])+'_'+str(i)+'.jpg',images[i])
      res_logits = sess.run(logits,feed_dict={image_placeholder:images})
      top1 = computeScore(res_logits,labels)
      print (top1)
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


