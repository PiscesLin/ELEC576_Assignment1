import os
import time
import tensorflow as tf
import tensorflow as tf2

if(tf.__version__.split('.')[0]=='2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()    

# Load MNIST dataset
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

def weight_variable(shape, type="random", name=""):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    if type == "random":
        initial = tf.random.truncated_normal(shape, stddev=0.1)
        W = tf.Variable(initial)
    elif type == "Xavier":
        initial = tf2.initializers.GlorotUniform()
        W = tf.Variable(initial(shape=shape))
    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    b = tf.Variable(tf.constant(0.1, shape=shape))

    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    return h_conv

def max_pool_2x2(x):
    ''']
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return h_max

# -----additional function for main_B and main_C--------------------------

def create_Summaries(targetVar, targetName):
    tf.summary.scalar('mean/' + targetName, tf.math.reduce_mean(targetVar))
    tf.summary.scalar('stddev/' + targetName, tf.math.reduce_std(targetVar))
    tf.summary.scalar('max/' + targetName, tf.math.reduce_max(targetVar))
    tf.summary.scalar('min/' + targetName, tf.math.reduce_min(targetVar))
    tf.summary.histogram(targetName, targetVar)
    return None

def neuFun(inputX, W, b, type):
    if type == "conv2d":
        return conv2d(inputX, W) + b
    elif type == "matmul":
        return tf.matmul(inputX, W) + b

def actFun(z, type, opName=""):
    if type == "Tanh":
        return tf.nn.tanh(z)
    elif type == "Sigmoid":
        return tf.nn.sigmoid(z)
    elif type == "ReLU":
        return tf.nn.relu(z)
    elif type == "leaky-ReLU":
        return tf.nn.leaky_relu(z, 0.01)
    elif type == "Softmax":
        return tf.nn.softmax(z, name=opName)

# -----main_A------------------------------------
def main_A(x, y_, result_dir):

    max_step = 5500  # the maximum iterations. After max_step iterations, the training will stop no matter what
    batch_size = 50  # set the batch size
    # reshape the input image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_conv')

    # FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING

    # setup training
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar(cross_entropy.op.name, cross_entropy)
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)

    # run the training
    for i in range(max_step):
        batch = mnist.train.next_batch(batch_size)  # make the data batch, which is used in the training iteration.
        # the batch size is 50
        if i % 100 == 0:
            # output the training accuracy every 100 iterations
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

            # Update the events file which is used to monitor the training (in this case,
            # only the training loss is monitored)
            summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

        # save the checkpoints every 1100 iterations
        if i % 1100 == 0 or i == max_step:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # run one train_step

    # print test error
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# -----main_B------------------------------------
def main_B(x, y_, result_dir):

    max_step = 5500  # the maximum iterations. After max_step iterations, the training will stop no matter what
    batch_size = 50  # set the batch size
    # reshape the input image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax
    W_fc2, b_fc2 = weight_bias([1024, 10], 10)
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_conv')

    # FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING

    # setup training
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    val_accuracy_ = tf.placeholder(tf.float32, name="val_accuracy")
    test_accuracy_ = tf.placeholder(tf.float32, name="test_accuracy")

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar(cross_entropy.op.name, cross_entropy)

    # Build the summary operation based on the TF collection of Summaries.
    create_Summaries(W_conv1, "Weight_conv1")
    create_Summaries(b_conv1, "bias_conv1")
    create_Summaries(h_conv1, "h_conv1")
    create_Summaries(h_pool1, "h_pool1")
    create_Summaries(W_conv2, "Weight_conv2")
    create_Summaries(b_conv2, "bias_conv2")
    create_Summaries(h_conv2, "h_conv2")
    create_Summaries(h_pool2, "h_pool2")
    create_Summaries(W_fc1, "Weight_fc1")
    create_Summaries(b_fc1, "bias_fc1")
    create_Summaries(h_fc1, "activation_fc1")
    create_Summaries(W_fc2, "Weight_fc2")
    create_Summaries(b_fc2, "bias_fc2")
    create_Summaries(y_conv, "activation_fc2")

    summary_op = tf.summary.merge_all()
    summary_op_test = tf.summary.scalar('test_accuracy', test_accuracy_)
    summary_op_val = tf.summary.scalar('val_accuracy', val_accuracy_)

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
    summary_writer_test = tf.summary.FileWriter(result_dir + '/test', sess.graph)
    summary_writer_val = tf.summary.FileWriter(result_dir + '/val', sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)

    # run the training
    for i in range(max_step):
        batch = mnist.train.next_batch(batch_size)  # make the data batch, which is used in the training iteration.
        # the batch size is 50
        if i % 100 == 0:
            # output the training accuracy every 100 iterations
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            # Update the events file which is used to monitor the training (in this case,
            # only the training loss is monitored)
            summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

        # save the checkpoints every 1100 iterations
        if i % 1100 == 0 or i == max_step:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)

            # print test accuracy
            test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            print("step %d, test accuracy %g" % (i, test_accuracy))
            summary_str_test = sess.run(summary_op_test, feed_dict={test_accuracy_: test_accuracy})
            summary_writer_test.add_summary(summary_str_test, i)
            summary_writer_test.flush()

            # print validation accuracy
            val_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
            print("step %d, validation accuracy %g" % (i, val_accuracy))
            summary_str_val = sess.run(summary_op_val, feed_dict={val_accuracy_: val_accuracy})
            summary_writer_val.add_summary(summary_str_val, i)
            summary_writer_val.flush()

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # run one train_step

    # print test error
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# -----main_C------------------------------------
def main_C(x, y_, result_dir):

    max_step = 5500  # the maximum iterations. After max_step iterations, the training will stop no matter what
    batch_size = 50  # set the batch size

    # different nonlinearities
    act_Type_List = ["Tanh", "Sigmoid", "ReLU", "leaky-ReLU"]
    act_Type = act_Type_List[3]

    # different initialization techniques
    initialize_Type_List = ["random", "Xavier"]
    initialize_Type = initialize_Type_List[0]

    # reshape the input image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32], type=initialize_Type)
    b_conv1 = bias_variable([32])
    neu1 = neuFun(x_image, W_conv1, b_conv1, "conv2d")
    h_conv1 = actFun(neu1, type=act_Type)
    h_pool1 = max_pool_2x2(h_conv1)

    # second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64], type=initialize_Type)
    b_conv2 = bias_variable([64])
    neu2 = neuFun(h_pool1, W_conv2, b_conv2, "conv2d")
    h_conv2 = actFun(neu2, type=act_Type)
    h_pool2 = max_pool_2x2(h_conv2)

    # densely connected layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024], type=initialize_Type)
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    z_fc1 = neuFun(h_pool2_flat, W_fc1, b_fc1, "matmul")
    h_fc1 = actFun(z_fc1, act_Type)

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax
    #W_fc2, b_fc2 = weight_bias([1024, 10], 10, initialType=initialize_Type)
    W_fc2 = weight_variable([1024, 10], type=initialize_Type)
    b_fc2 = bias_variable([10])
    z_fc2 = neuFun(h_fc1_drop, W_fc2, b_fc2, "matmul")
    y_conv = actFun(z_fc2, "Softmax", opName="y_conv")

    # FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING

    # setup training
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    # different training algorithms(Adagrad & Adam)
    # train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # Adam algorithms is better
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    val_accuracy_ = tf.placeholder(tf.float32, name="val_accuracy")
    test_accuracy_ = tf.placeholder(tf.float32, name="test_accuracy")

    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar(cross_entropy.op.name, cross_entropy)

    # Build the summary operation based on the TF collection of Summaries.
    create_Summaries(W_conv1, "Weight_conv1")
    create_Summaries(b_conv1, "bias_conv1")
    create_Summaries(h_conv1, "h_conv1")
    create_Summaries(h_pool1, "h_pool1")
    create_Summaries(W_conv2, "Weight_conv2")
    create_Summaries(b_conv2, "bias_conv2")
    create_Summaries(h_conv2, "h_conv2")
    create_Summaries(h_pool2, "h_pool2")
    create_Summaries(W_fc1, "Weight_fc1")
    create_Summaries(b_fc1, "bias_fc1")
    create_Summaries(h_fc1, "activation_fc1")
    create_Summaries(W_fc2, "Weight_fc2")
    create_Summaries(b_fc2, "bias_fc2")
    create_Summaries(y_conv, "activation_fc2")
    summary_op = tf.summary.merge_all()
    summary_op_test = tf.summary.scalar('test_accuracy', test_accuracy_)
    summary_op_val = tf.summary.scalar('val_accuracy', val_accuracy_)

    # Add the variable initializer Op.
    init = tf.initialize_all_variables()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)
    summary_writer_test = tf.summary.FileWriter(result_dir + '/test', sess.graph)
    summary_writer_val = tf.summary.FileWriter(result_dir + '/val', sess.graph)

    # Run the Op to initialize the variables.
    sess.run(init)

    # run the training
    for i in range(max_step):
        batch = mnist.train.next_batch(batch_size)  # make the data batch, which is used in the training iteration.
        # the batch size is 50
        if i % 100 == 0:
            # output the training accuracy every 100 iterations
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
            # Update the events file which is used to monitor the training (in this case,
            # only the training loss is monitored)
            summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()

        # save the checkpoints every 1100 iterations
        if i % 1100 == 0 or i == max_step:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)

            # print test accuracy
            test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            print("step %d, test accuracy %g" % (i, test_accuracy))
            summary_str_test = sess.run(summary_op_test, feed_dict={test_accuracy_: test_accuracy})
            summary_writer_test.add_summary(summary_str_test, i)
            summary_writer_test.flush()

            # print validation accuracy
            val_accuracy = accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
            print("step %d, validation accuracy %g" % (i, val_accuracy))
            summary_str_val = sess.run(summary_op_val, feed_dict={val_accuracy_: val_accuracy})
            summary_writer_val.add_summary(summary_str_val, i)
            summary_writer_val.flush()

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})  # run one train_step

    # print test error
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# -----------------------------------------------

# -----main_Function: Select which function you want to run here-------------

def main():
    start_time = time.time()  # start timing

    # placeholders for input data and input labeles
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

    # uncomment to select the program you want to run

    # result_dir = './A_results/'  # directory where the results for part A from the training are saved
    # main_A(x, y_, result_dir)

    # result_dir = './B_results/'  # directory where the results for part B from the training are saved
    # main_B(x, y_, result_dir)

    result_dir = './C_results/'  # directory where the results for part C from the training are saved
    main_C(x, y_, result_dir)

    stop_time = time.time()
    print('The training takes %f seconds to finish' % (stop_time - start_time))

# ///////////////////////////////////////////////////////////////

if __name__ == "__main__":
    main()

