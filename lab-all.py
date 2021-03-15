#--coding='utf-8'--
#start act 2
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data
x= tf.placeholder(tf.float32, [None, 784])
y_= tf.placeholder(tf.float32, [None, 10])
def weight_variable(shape,name1):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=name1)
def bias_variable(shape,name1):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name1)
W01 = weight_variable([784, 100],'W01')
b01 = bias_variable([100],'b01')
h01=tf.matmul(x,W01) + b01
x01 = tf.nn.relu(h01)
#end act 2
#start act 3
W02 = weight_variable([100,50],'W02')
b02 = bias_variable([50],'b02')
x02 = tf.nn.relu(tf.matmul(x01, W02) + b02)
W03 = weight_variable([50,10],'W03')
b03 = bias_variable([10],'b03')
h03 = tf.matmul(x02, W03) + b03
y=tf.nn.softmax(h03) # ?y=tf.nn.softmax(h02,1)
#end act 3
#start act 4
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *tf.log(y),reduction_indices=[1]))
train_step =   tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#end act 4
#start act 5
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
mnist = input_data.read_data_sets('/data/MNIST_data',one_hot=True)

for i in range(4000):
    batch_xs,batch_ys = mnist.train.next_batch(80)
    sess.run(train_step, feed_dict={x: batch_xs,y_: batch_ys})
    if i%200 ==0:
        print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))
    W10,b10,W20,b20,W30,b30 = sess.run([W01,b01,W02,b02,W03,b03])
    with open('/data/wb.data','wb') as f:
        pickle.dump([W10,b10,W20,b20,W30,b30],f)
#end act 5
python3 /data/MNIST_FC.py
