from .cnn_model import *


def run_cnn():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    height = 28
    width = 28
    num_channel = 1
    num_classes = 10

    # None은 지정되지 않은 값. Batch Size는 아직 정하지 않았기 때문에 모르는 값으로 넣어준다
    x = tf.placeholder('float', shape=[None, height, width, num_channel], name='data_input')
    y = tf.placeholder('float', shape=[None, num_classes], name='data_output')

    pred = cnn(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    optm = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

    corr = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, 1))
    accr = tf.reduce_mean(tf.cast(corr, 'float'))

    init = tf.global_variables_initializer()
    print("Functions Ready~!!!")

    num_epochs = 20
    batch_size = 64
    display_step = 1

    with tf.Session() as sess:
        sess.run(init)

        num_batches = int(len(x_train) // batch_size)
        for epoch in range(num_epochs):
            accr_total = 0.

            trainset = [(x.reshape(height, width, 1), onehot_encoder(y)) for x, y in zip(x_train, y_train)]
            np.random.shuffle(trainset)
            for step in range(num_batches):
                train_batch = trainset[step * batch_size : (step + 1) * batch_size]
                train_x = []
                train_y = []
                for batches in train_batch:
                    train_x.append(batches[0])
                    train_y.append(batches[1])

                feed_dict = {x: np.array(train_x), y:np.array(train_y)}
                _, accr_value = sess.run([optm, accr], feed_dict)

                accr_total += accr_value

            accuracy = accr_total / num_batches
            print('Accuracy of Epoch %d is %.4f' % (epoch, accuracy))