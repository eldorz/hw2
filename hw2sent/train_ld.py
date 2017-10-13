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
import winsound

import implementation as imp

batch_size = imp.batch_size
iterations = 100000
seq_length = 40  # Maximum length of sentence

checkpoints_dir = "./checkpoints_final"

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
glove_array[10] = 500000  # test out of bounds index
input_data, labels, optimizer, accuracy, loss, dropout_on, dropout_off = \
    imp.define_graph(glove_array)

# tensorboard
train_accuracy_op = tf.summary.scalar("training_accuracy", accuracy)
test_accuracy_op = tf.summary.scalar("test_accuracy", accuracy)
loss_op = tf.summary.scalar("loss", loss)
histograms = [tf.summary.histogram(var.op.name, var) for 
    var in tf.trainable_variables()]
summary_op = tf.summary.merge([train_accuracy_op, loss_op, histograms])

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
smoothed_train_acc = 0.5

sess.run(dropout_on)

for i in range(iterations + 1):
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
        sess.run(dropout_off)
        test_acc, test_summ = sess.run(
            [accuracy, test_accuracy_op],
            {input_data: test_data,
             labels: test_labels})
        sess.run(dropout_on)

        writer.add_summary(test_summ, i)

        print()
        print("Iteration: ", i)
        print("loss", loss_value)
        print("acc", accuracy_value)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        smoothed_train_acc = alpha * smoothed_train_acc + (1 - alpha) * accuracy_value
        print("smoothed train accuracy", smoothed_train_acc)
        print("test acc", test_acc)
        smoothed_acc = alpha * smoothed_acc + (1 - alpha) * test_acc
        print("smoothed accuracy", smoothed_acc)
        
        if smoothed_acc > best_smooth_acc:
            best_smooth_acc = smoothed_acc
            best_i = i
            if best_smooth_acc > 0.76:
                if not os.path.exists(checkpoints_dir):
                    os.makedirs(checkpoints_dir)
                save_path = all_saver.save(sess, checkpoints_dir +
                    "/trained_model.ckpt", global_step=i)
                print("Saved model to %s" % save_path)
                file = open("log.txt", "a")
                file.write("{0} {1}".format(best_smooth_acc, i) + "\n")
                file.close()

        if  smoothed_train_acc > 0.90:
            break
        print("best smoothed accuracy", best_smooth_acc)

    if (i % 10000 == 0 and i != 0):
        file = open("log.txt", "a")
        file.write("smoothed accuracy at {0} is {1}".format(i, smoothed_acc) + "\n")
        file.close()

    #if (i > 5000 and best_smooth_acc < 0.6): break

# write best performance to file       
file = open("log.txt", "a")
file.write("best smoothed accuracy   : {0}".format(best_smooth_acc) + "\n")
file.write("at iteration             : {0}".format(best_i) + "\n\n")
file.close()

sess.close()

Freq = 1000
Dur = 500
winsound.Beep(Freq, Dur)