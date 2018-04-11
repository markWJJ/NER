import tensorflow as tf

# t=tf.Variable(tf.random_uniform(shape=[5,10]))
s=tf.ones([5,])
s=s+1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    ss=sess.run(s)
    print(ss)