import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def train():
    mnist = input_data.read_data_sets("MINIST_data", one_hot=True)

    """ 测试数据集 
    print(mnist.train.images.shape, mnist.train.labels.shape)
    print(mnist.test.images.shape, mnist.test.labels.shape)
    print(mnist.validation.images.shape, mnist.validation.labels.shape)
    """
    sess = tf.InteractiveSession()
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 50 == 0:
            print(i, sess.run(W), sess.run(b))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def main():
    train()


if __name__ == '__main__':
    main()
