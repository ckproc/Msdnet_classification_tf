"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import h5py
import math
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
#import models.msdnet as msdnet
#import cifar
import msdnet
'''
def testontest():
    num_examples=10000
    num_iter = int(math.ceil(num_examples / args.batch_size))
    #true_count = 0  # Counts the number of correct predictions.
    top1_count=0
    top5_count=0
    test_step=0
    total_sample_count = num_iter * args.batch_size
    while test_step<num_iter:
     #predictions,targets = sess.run([top_k_op,labels],{image_paths_placeholder: testfilenames})
     predictions,targets = sess.run([top_k_op,labels_test])
     #print (targets.shape[0])
     for i in range(targets.shape[0]):
      if targets[i] in predictions.indices[i]:
       top5_count += 1
      if targets[i] == predictions.indices[i][0]:
       top1_count += 1
     #top1_count += np.sum(predictions)
     #top5_count += np.sum(predictions)
     test_step += 1
     precision1=top1_count/total_sample_count
     precision5=top5_count/total_sample_count
     print ('total test step:',test_step)
     print('%s: precision @ 1 = %.3f' % (datetime.now(), precision1),'%s: precision @ 5 = %.3f' % (datetime.now(), precision5))
     with open('test_result.txt','at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (test_step, precision1, precision5)) 
'''

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

    
def computeScore(prediction,labels_test):
    #32*10
    top5_count = 0
    top1_count = 0
    batchsize = prediction.shape[0]
    for i in range(batchsize):
      sort_indices = np.argsort(prediction[i])
      #print (labels_test[i].eval(),"|",sort_indices)
      if labels_test[i] in sort_indices[0:5]:
        top5_count += 1
      if labels_test[i] != sort_indices[-1]:
        top1_count += 1
    return top1_count,top5_count
    
def main(args):
  
    #network = importlib.import_module(args.model_def)   
    #trainfilenames = [os.path.join(args.data_dir, 'data_batch_%d.bin' % i) for i in xrange(1, 6)]
    #testfilenames = [os.path.join(data_dir, 'test_batch.bin')]
    args.grFactor = args.grFactor.split('-')
    args.bnFactor = args.bnFactor.split('-')
    
    for i in range(len(args.grFactor)):
        args.grFactor[i]=int(args.grFactor[i])
    for i in range(len(args.bnFactor)):
        args.bnFactor[i]=int(args.bnFactor[i])   
    with tf.Graph().as_default():
     #with tf.device('/gpu:1'):
      with tf.variable_scope("model") as scope:
        global_step = tf.Variable(0, trainable=False)
        #global_step = tf.train.get_or_create_global_step()
        #image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        #with tf.device('/cpu:0'):
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        images, labels=msdnet.distorted_inputs(args,batch_size_placeholder)
        images = tf.identity(images, 'input')
        
        #images_test,labels_test=msdnet.inputs(args, True)
        #inference model
        logits,end_points = msdnet.inference(images,args,phase_train_placeholder)   #logits shape[10,32,10]
        #print (images)
        #print (end_points)
        #list = []
        #print (len(end_points))
        #i=0
        #for item in end_points:
        #  sublist=[]
        #  if i%2==0:
        #   for tensor in item:
        #    sublist.append(tensor.name)
        #  else:
        #   sublist.append(item.name)
        #  list.append(sublist)
        #  i+=1
        #print (list)
        #print (np.shape(logits))
        loss = msdnet.multi_loss(logits,labels)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        #l2_loss=tf.add_n(regularization_losses)
        total_loss = tf.add_n(loss + regularization_losses, name='total_loss')
        train_op = msdnet.train(total_loss,global_step,args)
        ###scope.reuse_variables()
        ###logits_test,_ = msdnet.inference(images_test,args)
        #top_k_op = tf.nn.in_top_k(logits_test, labels_test,5)
        #top_k_op = tf.nn.top_k(logits_test, 5)
        
        #calculate loss
        
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        
        saver = tf.train.Saver(tf.global_variables(),max_to_keep = 50)
        #merged_summary_op = tf.merge_all_summaries()
        summary_op = tf.summary.merge_all()
        #saver = tf.train.Saver(tf.global_variables())
        init = tf.global_variables_initializer()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=args.log_device_placement))
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        #summary_writer = tf.train.SummaryWriter(args.logs, sess.graph)
        summary_writer = tf.summary.FileWriter(args.logs, sess.graph)
        best = 100
        with sess.as_default():
            for step in xrange(args.max_steps):
                start_time = time.time()
                #_, loss_value = sess.run([train_op, loss],{image_paths_placeholder: trainfilenames})
                _, loss_value= sess.run([train_op, loss],feed_dict={batch_size_placeholder:args.batch_size,phase_train_placeholder:True})
                loss_value = np.array(loss_value).sum()/float(len(loss_value))
                duration = time.time() - start_time
                #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                if step % 10 == 0:
                    num_examples_per_step = args.batch_size * args.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / args.num_gpus

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                    print (format_str % (datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch))
                    #testontest(sess, args, )
                #if step % 100 == 0:
                #    summary_str = sess.run(summary_op)
                #    summary_writer.add_summary(summary_str, global_step=step)

                # Save the model checkpoint periodically.
                
                if (step % 703 == 0 or (step + 1) == args.max_steps) and step !=0:
                    epoch  = step/703
                    num_examples=10000
                    num_iter = int(math.ceil(num_examples / args.batch_size))
                    #top1_count=0
                    #top5_count=0
                    test_step=0
                    N = 0.0
                    total_sample_count = num_iter * args.batch_size
                    top1All=[0.0]*10
                    top5All=[0.0]*10
                    top1Evolve=[0.0]*10
                    top5Evolve=[0.0]*10
                    #while test_step<num_iter and not coord.should_stop():
                    #while test_step<num_iter:
                    while False:
                      #predictions,targets = sess.run([top_k_op,labels],{image_paths_placeholder: testfilenames})
                      #predictions[10*32*10]       
                      #targets[32*10] 
                     predictions,targets = sess.run([logits_test,labels_test])
                     ensemble = np.zeros( (args.batch_size,10),dtype = float )
                     for i in range(len(predictions)):
                       top1,top5 = computeScore(predictions[i],targets)
                       top1All[i] = top1All[i] + top1
                       top5All[i] = top5All[i] + top5
                       ensemble = ensemble+softmax(predictions[i])
                       top1, top5 = computeScore(ensemble, targets)
                       top1Evolve[i] = top1Evolve[i] + top1
                       top5Evolve[i] = top5Evolve[i] + top5
                     N=N+args.batch_size
                     #print (('step %d, top1 %.3f (cumul: %.3f), top5 %.3f(cumul: %.3f)') %(test_step,top1,top1All[args.nBlocks-1]/N
                     #                 ,top5,top5All[args.nBlocks-1]/N)) # here
                     test_step += 1
                     
                    #for i in range(len(top1All)):
                    #    top1All[i] = top1All[i] / N
                    #    top5All[i] = top5All[i] / N
                    #    top1Evolve[i] = top1Evolve[i] / N
                    #    top5Evolve[i] = top5Evolve[i] / N
                    #    print (('%d exit top1: %.3f  top5: %.3f, \t Ensemble %d exit(s) top1: %.3f  top5: %.3f') 
                    #                       %(i,top1All[i],top5All[i],i,top1Evolve[i],top5Evolve[i]))
                                           
                    #with open('single_result.txt','at') as f:
                    #    f.write('%d\t' %epoch)
                    #    f.write(str(top1All)+'\n')
                    #with open('ensemble_result.txt','at') as f:
                    #    f.write('%d\t' %epoch)
                    #    f.write(str(top1Evolve)+'\n')
                    
                    
                    #if best-top1All[9]>0:
                    #  best = top1All[9]
                    #  print ('epoch %d'%epoch,'best err:',best)
                    print('Saving variables')
                    start_time = time.time()
                    checkpoint_path = os.path.join(args.train_dir, 'model-%s.ckpt' % str(step))
                      #checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
                    saver.save(sess, checkpoint_path, global_step=step,write_meta_graph=False)
                    save_time_variables = time.time() - start_time
                    print('Variables saved in %.2f seconds' % save_time_variables)
                      #save graph
                    metagraph_filename = os.path.join(args.train_dir, 'model.meta')
                    if not os.path.exists(metagraph_filename):
                        print('Saving metagraph')
                        saver.export_meta_graph(metagraph_filename)
                        save_time_metagraph = time.time() - start_time
                        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
                if(False):
                    print('Saving variables')
                    start_time = time.time()
                    checkpoint_path = os.path.join(args.train_dir, 'model-%s.ckpt' % str(step))
                    #checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
                    saver.save(sess, checkpoint_path, global_step=step,write_meta_graph=False)
                    save_time_variables = time.time() - start_time
                    print('Variables saved in %.2f seconds' % save_time_variables)
                    #save graph
                    metagraph_filename = os.path.join(args.train_dir, 'model.meta')
                    if not os.path.exists(metagraph_filename):
                        print('Saving metagraph')
                        saver.export_meta_graph(metagraph_filename)
                        save_time_metagraph = time.time() - start_time
                        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
                    
        
    
  
  

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
