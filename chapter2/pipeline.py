import tensorflow as tf
import numpy as np
import os

graph = tf.Graph()
session = tf.InteractiveSession(graph=graph)

filenames = ['test%d.txt' % i for i in range(1,4)]
filename_queue = tf.train.string_input_producer(filenames,capacity=3,shuffle=True,name='string_input_producer')

for f in filenames:
    if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)
    else:
        print('File %s found.' %f)
    
reader = tf.TextLineReader()

key, value = reader.read(filename_queue,name='text_read_op')

record_defaults = [[-1.0],[-1.0],[-1.0],[-1.0],[-1.0],[-1.0],[-1.0],[-1.0],[-1.0],[-1.0]]

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = tf.decode_csv(value, record_defaults=record_defaults)

features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10])

x = tf.train.shuffle_batch([features], batch_size=3, capacity=5, name='data_batch',min_after_dequeue=1,num_threads=1)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=session)

W = tf.Variable(tf.random_uniform(shape=[10,5],minval=-0.1,maxval=0.1,dtype=tf.float32),name='W')
b = tf.Variable(tf.zeros([5],dtype=tf.float32),name='b')
h = tf.nn.softmax(tf.matmul(x,W) + b)

tf.global_variables_initializer().run()

for step in range(10):
    x_eval, h_eval = session.run([x,h])
    print("============ step %d ============" % step)
    print('Evaluated data (x)')
    print(x_eval)
    print('Evaluated data (h)')
    print(h_eval)
    print('')

coord.request_stop()
coord.join(threads)
session.close()