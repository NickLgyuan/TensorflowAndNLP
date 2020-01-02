import tensorflow as tf
import numpy as np

graph = tf.Graph()
session = tf.InteractiveSession(graph=graph)

x = tf.placeholder(shape=[1,10],dtype=tf.float32,name='x')
# x = tf.constant(value=[[]],dtype=tf.float32,name='x')
W = tf.Variable(tf.random_uniform(shape=[10,5],minval=-0.1,maxval=0.1, dtype=tf.float32), name='W')
b = tf.Variable(tf.zeros(shape=[5],dtype=tf.float32), name='b')
h = tf.nn.sigmoid(tf.matmul(x,W) + b)

tf.global_variables_initializer().run()
h_eval = session.run(h,feed_dict={x:np.random.rand(1,10)})
print(h_eval)
session.close()
