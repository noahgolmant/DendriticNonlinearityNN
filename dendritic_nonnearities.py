from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

import tensorflow as tf

EPOCHS = 15
LEARNING_RATE = 0.001
BATCH_SIZE = 100
DISPLAY_STEP = 1

NUM_CLASSES = 10
BRANCH_SIZE = 10
INPUT_SIZE = 784

x = tf.placeholder("float", [None, INPUT_SIZE])
y = tf.placeholder("float", [None, NUM_CLASSES])

# initialize wieghts, biases...
#weights = tf.placeholder("float", [INPUT_SIZE, BRANCH_SIZE]) # N x B
#biases = tf.placeholder("float", [INPUT_SIZE, BRANCH_SIZE])  # N x B
weights = tf.Variable(tf.random_normal([INPUT_SIZE, BRANCH_SIZE]))
biases = tf.Variable(tf.random_normal([INPUT_SIZE, BRANCH_SIZE]))

def dendritic_nonlinearity_net(x, weights, biases):
    print(x.get_shape())
    A = tf.mul(weights, x[:, :, None]) + biases
    print(A.get_shape())
    Y = tf.tanh(A)
    Z = tf.reduce_prod(Y, reduction_indices=[1])
    print(Z.get_shape())
    out = tf.nn.softmax(Z) # figure this out... softmax? just reduce by sum and then sigmoid?
    return out

pred = dendritic_nonlinearity_net(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    print("Beginning learning...")
    sess.run(init)
    for epoch in range(EPOCHS):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/BATCH_SIZE)

        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            _, c = sess.run([optimizer, cost], feed_dict = {
                x: batch_x,
                y: batch_y
            })

            avg_cost += c / total_batch
        if epoch % DISPLAY_STEP == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))
    print("Finished optimization.")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Test accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))