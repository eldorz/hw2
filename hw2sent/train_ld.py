"""
You are encouraged to edit this file during development, however your final
model must be trained using the original version of this file. This file
trains the model defined in implementation.py, performs tensorboard logging,
and saves the model to disk every 10000 iterations. It also prints loss
values to stdout every 50 iterations.
"""


import numpy as np
import tensorflow as tf
from random import randint
import datetime
import os

import implementation as imp

batch_size = imp.batch_size
iterations = 30000
seq_length = 40  # Maximum length of sentence

checkpoints_dir = "./checkpoints"

def getTrainBatch():
    labels = []
    arr = np.zeros([batch_size, seq_length])
    for i in range(batch_size):
        if (i % 2 == 0):
            num = randint(0, 9999)
            labels.append([1, 0])
        else:
            num = randint(12500, 22499)
            labels.append([0, 1])
        arr[i] = training_data[num]
    return arr, labels

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

# Call implementation
glove_array, glove_dict = imp.load_glove_embeddings()
training_data = imp.load_data(glove_dict)
input_data, labels, optimizer, accuracy, loss = imp.define_graph(glove_array)

# tensorboard
train_accuracy_op = tf.summary.scalar("training_accuracy", accuracy)
test_accuracy_op = tf.summary.scalar("test_accuracy", accuracy)
loss_op = tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge([train_accuracy_op, loss_op])

# saver
all_saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

logdir = "tensorboard/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

best_test_acc = 0
best_i = 0
alpha = 0.95
smoothed_acc = 0.5
best_smooth_acc = 0

for i in range(iterations):
    batch_data, batch_labels = getTrainBatch()
    sess.run(optimizer, {input_data: batch_data, labels: batch_labels})
    if (i % 50 == 0):
        loss_value, accuracy_value, summary = sess.run(
            [loss, accuracy, summary_op],
            {input_data: batch_data,
             labels: batch_labels})
        writer.add_summary(summary, i)

        # run on test data
        test_data, test_labels = getValidBatch()
        test_acc, test_summ = sess.run(
            [accuracy, test_accuracy_op],
            {input_data: test_data,
             labels: test_labels})
        writer.add_summary(test_summ, i)

        print()
        print("Iteration: ", i)
        print("loss", loss_value)
        print("acc", accuracy_value)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print("test acc", test_acc)
        print("best test acc", best_test_acc, "at timestep", best_i)
        smoothed_acc = alpha * smoothed_acc + (1 - alpha) * test_acc
        print("smoothed accuracy", smoothed_acc)
        if smoothed_acc > best_smooth_acc:
            best_smooth_acc = smoothed_acc
            best_i = i

    if (i % 10000 == 0 and i != 0):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        save_path = all_saver.save(sess, logdir +
                                   "/model.ckpt",
                                   global_step=i)
        print("Saved model to %s" % save_path)

# write best performance to file       
file = open("log.txt", "w")
file.write(best_smooth_acc + " " + best_i)
file.write("\n")
file.close()

sess.close()
