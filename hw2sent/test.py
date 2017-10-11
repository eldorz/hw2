import numpy as np
import tensorflow as tf 
import implementation as imp
from random import randint

checkpoints_dir = "./checkpoints"
batch_size = 30
seq_length = 40
num_batches = 1000

def getValidBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(10000, 12499)
            labels.append([1, 0])
        else:
            num = randint(22500, 24999)
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels

glove_array, glove_dict = imp.load_glove_embeddings()
training_data = imp.load_data(glove_dict)
input_data, labels, dropout_keep_prob, optimizer, accuracy, loss = \
    imp.define_graph(glove_array)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Restore variables from disk.
    saver.restore(sess, checkpoints_dir + "/trained_model.ckpt-30000")
    print("Model restored.")

    sess.run(dropout_keep_prob.assign(1.0))

    tot = float(0.0)
    for i in range(num_batches):
        test_data, test_labels = getValidBatch()
        test_acc = sess.run(accuracy, {input_data: test_data, labels: test_labels})
        tot += test_acc
        print("accuracy on test set %d: %f" % (i, test_acc))
    print("average accuracy: %f" % (tot / float(num_batches)))
