import tensorflow as tf
import numpy as np
from PIL import Image
import os, csv

PATH = os.path.dirname('__file__')  # Current path
csv_path = os.path.join(PATH, 'gt_training.csv')
csv_file = [i for i in csv.reader(open(csv_path))]

def GET_LABLE(m_id):
    VECTOR = np.eye(6)  # 6*6 Unit diagonal matrix
    classes = ['Homogeneous', 'Speckled', 'Nucleolar', 'Centromere', 'NuMem', 'Golgi']
    for i in csv_file:
        try:
            if m_id == int(i[0]):
                return VECTOR[classes.index(i[1])]
        except:
            pass
# Read images
Tr_path = os.path.join(PATH, 'training')
Tr_files = os.listdir(Tr_path)
T_path = os.path.join(PATH, 'test')
T_files = os.listdir(T_path)
V_path = os.path.join(PATH, 'validation')
V_files = os.listdir(V_path)

labels = set()
Tr_images, Tr_labels = [], []
for x in Tr_files:
    # Open file(image)
    path = os.path.join(Tr_path, x)
    m = Image.open(path)  # Read images
    # Reshape image to fit the network structure
    m = m.resize((78, 78))
    # Convert to numpy array and pre process the image
    m = np.asarray(m, dtype=np.float32)
    m = m[:, :, np.newaxis]  # 78*78*1，
    m = (m - np.min(m)) / (np.max(m) - np.min(m))  # normalization
    m -= 0.5  # Speed up
    # Image Rotation
    m90 = np.rot90(m, 1)
    m180 = np.rot90(m, 2)
    m270 = np.rot90(m, 3)
    # Get image id and label
    m_id = int(x.split('.')[0])
    label = GET_LABLE(m_id)

    Tr_images += [m, m90, m180, m270]
    Tr_labels += [label] * 4

T_images, T_labels = [], []
for x in T_files:
    # Open file
    path = os.path.join(T_path, x)
    m = Image.open(path)
    m = m.resize((78, 78))
    m = np.asarray(m, dtype=np.float32)
    m = m[:, :, np.newaxis]
    m = (m - np.min(m)) / (np.max(m) - np.min(m))
    m -= 0.5
    m90 = np.rot90(m, 1)
    m180 = np.rot90(m, 2)
    m270 = np.rot90(m, 3)
    m_id = int(x.split('.')[0])
    label = GET_LABLE(m_id)

    T_images += [m, m90, m180, m270]
    T_labels += [label] * 4

V_images, V_labels = [], []
for x in V_files:
    # Open file
    path = os.path.join(V_path, x)
    m = Image.open(path)
    m = m.resize((78, 78))
    m = np.asarray(m, dtype=np.float32)
    m = m[:, :, np.newaxis]
    m = (m - np.min(m)) / (np.max(m) - np.min(m))
    m -= 0.5
    m90 = np.rot90(m, 1)
    m180 = np.rot90(m, 2)
    m270 = np.rot90(m, 3)
    m_id = int(x.split('.')[0])
    label = GET_LABLE(m_id)

    V_images += [m, m90, m180, m270]
    V_labels += [label] * 4
print("Train set:", len(Tr_images), "Test set:", len(T_images), "Validation set:", len(V_images))

BATCH_SIZE = 120
learning_rate = 0.01

# Define functions
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

# Construct convolutional layer
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# Construct 2x2 max-pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Construct 3x3 max-pooling layer
def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

# Define model
def Construct(X, regularizer):
    # Define layer 1 conv1
    with tf.variable_scope('l1-conv1'):
        conv1_w = weight_variable([7, 7, 1, 6])
        conv1_b = bias_variable([6])
        conv1 = conv2d(X, conv1_w)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))

    # Define layer 2 pool1
    with tf.name_scope("l2_pool1"):
        pool1 = max_pool_2x2(relu1)

    # Define layer 3 conv2
    with tf.name_scope("l3_conv2"):
        conv2_w = weight_variable([4, 4, 6, 16])
        conv2_b = bias_variable([16])
        conv2 = conv2d(pool1, conv2_w)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))

    # Define layer 4 pool2
    with tf.name_scope("l4_pool2"):
        pool2 = max_pool_3x3(relu2)

    # Define layer 5 conv3
    with tf.name_scope("l5_conv3"):
        conv3_w = weight_variable([3, 3, 16, 32])
        conv3_b = bias_variable([32])
        conv3 = conv2d(pool2, conv3_w)
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_b))

    # Define layer 6 pool3
    with tf.name_scope("l6_pool3"):
        pool3 = max_pool_3x3(relu3)

    # flatten layer 7 for fully-connected layer
    faltten_pool3 = tf.contrib.layers.flatten(pool3)
    faltten_pool3_nodes = faltten_pool3.get_shape().as_list()[1]

    # Define layer 8 fc1
    with tf.variable_scope('l7-fc1'):
        fc1_w = weight_variable([faltten_pool3_nodes, 150])
        fc1_b = bias_variable([150])
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_w))
        fc1 = tf.nn.relu(tf.matmul(faltten_pool3, fc1_w) + fc1_b)
        fc1 = tf.nn.dropout(fc1, 0.5)

    # Define layer 9 fc2 (output layer)
    with tf.variable_scope('l8-fc2'):

        fc2_w = weight_variable([150, 6])
        fc2_b = bias_variable([6])
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_w))

        logit = tf.matmul(fc1, fc2_w) + fc2_b

    return logit

# Train the model
with tf.name_scope("input"):  # Define the format of placeholders, inputs, and outputs
    x = tf.placeholder(tf.float32, [None, 78, 78, 1], name='x')
    y_ = tf.placeholder(tf.float32, [None, 6], name='y-input')

# Accuracy
with tf.name_scope("accuracy"):
    regularizer = tf.contrib.layers.l2_regularizer(0.001)
    y = Construct(x, regularizer)
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# COST
with tf.name_scope("COST"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    COST = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

# optimizer
with tf.name_scope("train_step"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(COST)

saver = tf.train.Saver(max_to_keep=3)

# Define generator to get batch
def BATCH(data, label, batch_size):
    for start_index in range(0, len(data) - batch_size + 1, batch_size):
        slice_index = slice(start_index, start_index + batch_size)
        yield data[slice_index], label[slice_index]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(300):
        train_COST, train_acc, batch_num = 0, 0, 0
        if i % 5 == 0:
            print("Epoch", i)
        for data, label in BATCH(Tr_images, Tr_labels, BATCH_SIZE):
            _, err, acc = sess.run([optimizer, COST, accuracy], feed_dict={x: data, y_: label})
            train_COST += err
            train_acc += acc
            batch_num += 1
        if i % 5 == 0:
            print("Training accuracy: ", train_acc / batch_num)

        test_COST, test_acc, batch_num = 0, 0, 0
        for data, label in BATCH(T_images, T_labels, BATCH_SIZE):
            err, acc = sess.run([COST, accuracy], feed_dict={x: data, y_: label})
            test_COST += err
            test_acc += acc
            batch_num += 1
        if i % 5 == 0:
            print("Test accuracy: ", test_acc / batch_num)

        # Save model for each 100 epoches
        if (i + 1) % 100 == 0:
            saver.save(sess, 'model/hep2.ckpt')

    validation_COST, validation_acc, batch_num = 0, 0, 0
    for data, label in BATCH(V_images, V_labels, BATCH_SIZE):
        err, acc = sess.run([COST, accuracy], feed_dict={x: data, y_: label})
        validation_COST += err
        validation_acc += acc
        batch_num += 1
    print("validation COST:", validation_COST / batch_num)
    print("validation acc:", validation_acc / batch_num)
