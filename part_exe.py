import tensorflow as tf
import numpy as np
import time
import sys
with tf.device('/cpu:0'):
    a = tf.placeholder(tf.float32, shape=[])
    b = tf.placeholder(tf.float32, shape=[])
    c = tf.placeholder(tf.float32, shape=[])
    r1 = tf.add(a, b)
    r4 = tf.add(r1,r1)
    r2 = tf.multiply(r1,r1)
    r3 = tf.multiply(r4,r4)
    with tf.Session() as sess:
        s1 = time.time()
        h = sess.partial_run_setup([r2, r3], [a,b])
        s2 = time.time()
        res = sess.partial_run(h, r2,feed_dict={a:2,b:2})
        s3 = time.time()
        print (res)
        re = sess.partial_run(h,r3)
        s4 = time.time()
        print(re)
        print (s2-s1)
        print (s3-s2)
        print (s4-s3)
sys.exit()
with tf.Session() as sess:
    a = tf.placeholder(tf.float32, shape=[])
    #a = tf.constant([0.01]*10000)
    #b = tf.constant([0.02]*10000)
    b = tf.placeholder(tf.float32, shape=[])
    #c = tf.placeholder(tf.float32, shape=[])
    r1 = tf.add(a, b)
    #r1 = tf.tensordot(a,b,1)
    r2 = tf.multiply(r1, r1)
    r3 = tf.multiply(r1,r1)
    s1=time.time()
    #for i in range(100):
    h = sess.partial_run_setup([r1,r2, r3],[a,b])
    #s2=time.time()
    res = sess.partial_run(h,r1,feed_dict={a:1,b:2})
    print (res)
    res = sess.partial_run(h, r2)
    print (res)
    #s3=time.time()
    #print (res)
    res = sess.partial_run(h, r3)
    print (res)
    s4 = time.time()
    #for i in range(100):
    #res = sess.run(r2)
    #s5 = time.time()
    #print (s2-s1)
    #print (s3-s2)
    #print (s4-s1)
    
   
   #print (s5-s4)
    
'''
s=time.time()
for i in range(10000):
 sess.run(d)
 #sess.run(y)
 #res = sess.run(y)
duration = time.time() - s
print (duration)

s=time.time()
for i in range(10000):
 #sess.run(d)
 sess.run(y)
 #res = sess.run(y)
duration = time.time() - s
print (duration)

s=time.time()
for i in range(10000):
 sess.run(d)
 sess.run(y)
 #res = sess.run(y)
duration = time.time() - s
print (duration)
'''