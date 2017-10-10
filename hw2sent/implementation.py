import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import re
import string
import math
import random
import inspect
import time

# Using tensorflow 1.2.1

#        data
#          |
#         / \
#       CNN  RNN
#         \ /
#          |
#  fully connected classifier

# constants
GLOVE_DIM = 50
NUM_REVIEWS = 25000
WORDS_PER_REVIEW = 40

# global hyperparameters
batch_size = 30
GLOVE_MAX_VOCAB = 50000  # 400000 words in glove datasete
DROPOUT_KEEP_PROB = 0.8
LEARNING_RATE = 0.0005
L2_BETA = 0.0001
ADAM_EPSILON = 0.001

# CNN hyperparameters
CNN_FILTERS = 1
CNN_FILTERSIZE = 3
CNN_POOL_SIZE = (WORDS_PER_REVIEW, 1)
CNN_POOL_STRIDES = (WORDS_PER_REVIEW, 1)

# RNN hyperparameters
LSTM_SIZE = 50
RNN_LAYERS = 1

# binary classifier hyperparameters
BIN_CLASS_LAYERS = 1
BIN_CLASS_HIDDEN_SIZE = 128


file = open("log.txt", "a")
file.write("\n")
file.write(time.strftime("%c") + "\n")
file.write("global\n")
file.write("  batch_size            : {0}".format(batch_size) + "\n")
file.write("  GLOVE_MAX_VOCAB       : {0}".format(GLOVE_MAX_VOCAB) + "\n")
file.write("  DROPOUT_KEEP_PROB     : {0}".format(DROPOUT_KEEP_PROB) + "\n")
file.write("  LEARNING_RATE         : {0}".format(LEARNING_RATE) + "\n")
file.write("  L2_BETA               : {0}".format(L2_BETA) + "\n")
file.write("CNN\n")
file.write("  CNN_FILTERS           : {0}".format(CNN_FILTERS) + "\n")
file.write("  CNN_FILTERSIZE        : {0}".format(CNN_FILTERSIZE) + "\n")
file.write("RNN\n")
file.write("  LSTM_SIZE             : {0}".format(LSTM_SIZE) + "\n")
file.write("  RNN_LAYERS            : {0}".format(RNN_LAYERS) + "\n")
file.write("classifier\n")
file.write("  BIN_CLASS_LAYERS      : {0}".format(BIN_CLASS_LAYERS) + "\n")
file.write("  BIN_CLASS_HIDDEN_SIZE : {0}".format(BIN_CLASS_HIDDEN_SIZE) + "\n")
file.close()

def preprocess(rawstring):
    # stopwords
    stops = {'the', 'and', 'a', 'of', 'to', 'is', 'in', 'it', 'i', 'this',
    'that', 'br', 'was', 'as', 'for', 'with', 'but', 'on', 'you', 'are',
    'his', 'her', 'have', 'he', 'she', 'be', 'one', 'its', 'at', 'all', 'by',
    'an', 'they', 'from', 'who', 'so', 'just', 'or', 'about', 'has', 'if'}
    nobr = re.sub(r'<br>', ' ', rawstring)
    no_punct = ''.join(c for c in nobr if c not in string.punctuation)
    lower = no_punct.lower()
    words = lower.split()
    processed = []
    for w in words:
        if w in stops: continue
        processed.append(w)
    return processed


def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""

    filename = 'reviews.tar.gz'
    dir = os.path.dirname(__file__)

    # just load data if already there
    if os.path.exists(os.path.join(dir, 'data.npy')):
        print("using saved data, delete 'data.npy' to reprocess")
        data = np.load('data.npy')
        return data

    # untar
    if not os.path.exists(os.path.join(dir, 'reviews/')):
        with tarfile.open(filename, "r") as tarball:
            tarball.extractall(os.path.join(dir, 'reviews/'))

    # load and preprocess
    
    # debug
    my_word_list = set()

    file_list = glob.glob(os.path.join(dir, 'reviews/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir, 'reviews/neg/*')))
    assert(len(file_list) == NUM_REVIEWS)
    data = np.empty([NUM_REVIEWS, WORDS_PER_REVIEW], dtype=np.intp)
    filenum = 0
    for f in file_list:
        with open(f, "r", encoding='utf8') as openf:
            s = openf.read()
            words = preprocess(s)
            word_indices = []
            wordnum = 0
            for w in words:
                if wordnum >= WORDS_PER_REVIEW: break
                if w in glove_dict:
                    # add index of known word
                    word_indices.append(glove_dict[w]) 
                    #debug
                    my_word_list.add(w)
                else:
                    # add the index of the unknown word
                    word_indices.append(glove_dict['UNK'])
                wordnum += 1

            # zero padding
            if wordnum < WORDS_PER_REVIEW:
                for i in range(wordnum, WORDS_PER_REVIEW):
                    word_indices.append(0)

            data[filenum] = word_indices
        filenum += 1
        
    np.save("data", data)

    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """

    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    
    with open("glove.6B.50d.txt",'r',encoding="utf-8") as f:
        data = f.readlines()

    embeddings = np.empty([GLOVE_MAX_VOCAB,GLOVE_DIM], dtype=np.float32)
    word_index_dict = {}

    word_index_dict['UNK'] = 0
    embeddings[0] = np.zeros(GLOVE_DIM)

    n = 1
    for d in data:
        if n >= GLOVE_MAX_VOCAB:
            break
        elements = d.split()
        word_index_dict[elements[0]] = n
        embeddings[n] = elements[1:]
        n += 1

    return embeddings, word_index_dict

def lstm_cell(dropout_keep):
    cell = tf.contrib.rnn.BasicLSTMCell(LSTM_SIZE, forget_bias = 0.0, 
        state_is_tuple = True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, 
       input_keep_prob = DROPOUT_KEEP_PROB,
       output_keep_prob = 1.0)

    return cell

def onelayer(input_tensor):
    output = tf.layers.dense(input_tensor, 2, name = "bin_class_layer_1")
    return output

def twolayer(input_tensor, dropout_keep):
    layer_one_output = tf.layers.dense(input_tensor, BIN_CLASS_HIDDEN_SIZE, 
        name = "bin_class_layer_1", activation = tf.nn.relu)
    layer_one_output = tf.nn.dropout(layer_one_output, dropout_keep)
    output = tf.layers.dense(layer_one_output, 2, name = "bin_class_layer_2")
    return output

def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""

    dropout_keep = tf.get_variable("dropout_keep", dtype = tf.float32,
        initializer = tf.constant(DROPOUT_KEEP_PROB))
    dropout_off = dropout_keep.assign(1.0)
    dropout_on = dropout_keep.assign(DROPOUT_KEEP_PROB)

    input_data = tf.placeholder(tf.int32,
        shape = (batch_size, WORDS_PER_REVIEW), name = "input_data")
    labels = tf.placeholder(tf.int32, shape = (batch_size, 2), name = "labels")

    # substitute embeddings for word indices
    #embeddings = tf.constant(glove_embeddings_arr, name = "embeddings")
    embeddings = tf.Variable(glove_embeddings_arr, name = "embeddings")
    input_embeddings = tf.nn.embedding_lookup(embeddings, input_data, 
        name = "input_embeddings")

    # convolutional layer
    # conv in [batch, time, wordvec, 1]
    conv_in = tf.expand_dims(input_embeddings, 3)
    # conv out [batch, time, wordvec, filters]
    conv_out = tf.layers.conv2d(inputs = conv_in,
        filters = CNN_FILTERS,
        kernel_size = [CNN_FILTERSIZE, GLOVE_DIM],
        padding = 'same',
        activation = tf.nn.relu, name = 'conv2d')

    # max pooling [batch, 1, wordvec, filters]
    max_pool_out = tf.layers.max_pooling2d(conv_out,
        pool_size = CNN_POOL_SIZE, 
        strides = CNN_POOL_STRIDES,
        padding='valid', data_format='channels_last', name='max_pool')

    # turn to vector for later connected layer [batch, wordvec * filters]
    max_pool_out = tf.reshape(max_pool_out, [batch_size, -1])

    '''
    # bidirectional layer
    bidir_ouputs, output_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw = lstm_cell(dropout_keep),
        cell_bw = lstm_cell(dropout_keep),
        inputs = input_embeddings,
        dtype = tf.float32,
        sequence_length = tf.fill([batch_size], WORDS_PER_REVIEW))
    fused_bidir_output = tf.concat([bidir_ouputs[0], bidir_ouputs[1]], 2)
    '''

    # multilayer lstm cell
    stacked_lstm_cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell(dropout_keep) for _ in range(RNN_LAYERS)], 
        state_is_tuple = True)

    outputs, last_states = tf.nn.dynamic_rnn(
        cell = stacked_lstm_cell,
        dtype = tf.float32, 
        sequence_length = tf.fill([batch_size], WORDS_PER_REVIEW), 
        inputs = input_embeddings)

    # mean pool the outputs
    rnn_out = tf.reduce_mean(outputs, 1)

    # concatenate outputs
        # output = tf.reshape(tf.concat(outputs, 1), 
            # [batch_size, LSTM_SIZE * WORDS_PER_REVIEW])

    # take only the last output
        #outputlist = tf.unstack(outputs, axis = 1)
        #rnn_out = outputlist[WORDS_PER_REVIEW - 1]


    # concatenate rnn and convolutional outputs
    bin_class_input = tf.concat([max_pool_out, rnn_out], 1)
   
    # binary classifier input dropout
    bin_class_input = tf.nn.dropout(bin_class_input, dropout_keep, 
        name = "bin_class_input_dropout")

    # binary classifier
    if BIN_CLASS_LAYERS == 1:
        logits = onelayer(bin_class_input)
    else:
        logits = twolayer(bin_class_input, dropout_keep)
    
    # stats
    preds = tf.argmax(logits, 1, name = "predictions")
    label_argmax = tf.argmax(labels, 1, name = "label_argmax")
    correct = tf.equal(label_argmax, preds, name = "correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy")
    
    # cross-entropy loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels = labels, logits = logits, name = "softmax_cross_entropy")
    loss = tf.reduce_mean(cross_entropy, name = "loss")

    # L2 regularisation
    l2 = L2_BETA * sum(tf.nn.l2_loss(tf_var) 
        for tf_var in tf.trainable_variables() if not ("Bias" in tf_var.name))
    loss += l2

    # optimiser
    adam = tf.train.AdamOptimizer(LEARNING_RATE, epsilon = ADAM_EPSILON)
    optimizer = adam.minimize(loss)

    # modify returned values if called by my altered train.py
    frm = inspect.stack()[1]
    mod = inspect.getmodule(frm[0])
    if mod.__file__ == "train_ld.py":
        return input_data, labels, optimizer, accuracy, loss, dropout_on, dropout_off
    
    return input_data, labels, dropout_keep, optimizer, accuracy, loss



