import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import re
import string
import math
import random

# Using tensorflow 1.3.0

batch_size = 100
GLOVE_DIM = 50
GLOVE_MAX_VOCAB = 50000  # 400000 words in glove datasete
NUM_REVIEWS = 25000
WORDS_PER_REVIEW = 40

# global hyperparameters
DROPOUT_KEEP_PROB = 0.7
LEARNING_RATE = 0.005

# RNN hyperparameters
BASIC_RNN_SIZE = 16    # for bidirectional layer
LSTM_SIZE = 16
RNN_LAYERS = 4

# binary classifier hyperparameters
BIN_CLASS_LAYERS = 1
BIN_CLASS_HIDDEN_SIZE = 128
'''
# global hyperparameters
DROPOUT_KEEP_PROB = random.gauss(0.7, 0.2)
LEARNING_RATE = random.gauss(0.005, 0.002)

# RNN hyperparameters
LSTM_SIZE = max(2, int(random.gauss(20.0, 10.0)))
RNN_LAYERS = max(1, int(random.gauss(4.0, 3.0)))

# binary classifier hyperparameters
BIN_CLASS_LAYERS = random.randint(1, 2)
BIN_CLASS_HIDDEN_SIZE = max(2, int(random.gauss(100.0, 50.0)))
'''

file = open("log.txt", "a")
file.write("batch_size            : {0}".format(batch_size) + "\n")
file.write("GLOVE_MAX_VOCAB       : {0}".format(GLOVE_MAX_VOCAB) + "\n")
file.write("DROPOUT_KEEP_PROB     : {0}".format(DROPOUT_KEEP_PROB) + "\n")
file.write("LEARNING_RATE         : {0}".format(LEARNING_RATE) + "\n")
file.write("BASIC_RNN_SIZE        : {0}".format(BASIC_RNN_SIZE) + "\n")
file.write("LSTM_SIZE             : {0}".format(LSTM_SIZE) + "\n")
file.write("RNN_LAYERS            : {0}".format(RNN_LAYERS) + "\n")
file.write("BIN_CLASS_LAYERS      : {0}".format(BIN_CLASS_LAYERS) + "\n")
file.write("BIN_CLASS_HIDDEN_SIZE : {0}".format(BIN_CLASS_HIDDEN_SIZE) + "\n")
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

    # debug
    file = open("words.txt", "w")
    for w in my_word_list:
        file.write(w + " ")
    file.close()

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
    cell = tf.nn.rnn_cell.BasicLSTMCell(LSTM_SIZE, forget_bias = 0.0, 
        state_is_tuple = True)
    #cell = tf.nn.rnn_cell.GRUCell(LSTM_SIZE)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, 
        input_keep_prob = DROPOUT_KEEP_PROB,
        output_keep_prob = 1.0,
        state_keep_prob = 1.0)

    return cell

def simple_recurrent_cell(dropout_keep):
    cell = tf.nn.rnn_cell.BasicRNNCell(BASIC_RNN_SIZE)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob = DROPOUT_KEEP_PROB,
        output_keep_prob = 1.0)
    return cell

def onelayer(input_tensor, dropout_keep):
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
        initializer = tf.constant(1.0))
    dropout_off = dropout_keep.assign(1.0)
    dropout_on = dropout_keep.assign(DROPOUT_KEEP_PROB)

    input_data = tf.placeholder(tf.int32,
        shape = (batch_size, WORDS_PER_REVIEW), name = "input_data")
    labels = tf.placeholder(tf.int32, shape = (batch_size, 2), name = "labels")

    # substitute embeddings for word indices
    embeddings = tf.constant(glove_embeddings_arr, name = "embeddings")
    input_embeddings = tf.nn.embedding_lookup(embeddings, input_data, 
        name = "input_embeddings")

    # bidirectional layer
    bidir_ouputs, output_states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw = simple_recurrent_cell(dropout_keep),
        cell_bw = simple_recurrent_cell(dropout_keep),
        inputs = input_embeddings,
        dtype = tf.float32,
        sequence_length = tf.fill([batch_size], WORDS_PER_REVIEW))
    print(bidir_ouputs)
    fused_bidir_output = tf.concat([bidir_ouputs[0], bidir_ouputs[1]], 2)
    print(fused_bidir_output)

    # multilayer lstm cell
    stacked_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_cell(dropout_keep) for _ in range(RNN_LAYERS)], 
        state_is_tuple = True)

    outputs, last_states = tf.nn.dynamic_rnn(
        cell = stacked_lstm_cell,
        dtype = tf.float32, 
        sequence_length = tf.fill([batch_size], WORDS_PER_REVIEW), 
        inputs = fused_bidir_output)

    output = tf.reshape(tf.concat(outputs, 1), 
        [batch_size, LSTM_SIZE * WORDS_PER_REVIEW])

    # rnn to layer 1 dropout
    output = tf.nn.dropout(output, dropout_keep, 
        name = "rnn_to_layer_1_dropout")

    # binary classifier
    if BIN_CLASS_LAYERS == 1:
        logits = onelayer(output, dropout_keep)
    else:
        logits = twolayer(output, dropout_keep)
    
    # stats
    preds = tf.argmax(logits, 1, output_type = tf.int32, name = "predictions")
    label_argmax = tf.argmax(labels, 1, output_type = tf.int32, 
        name = "label_argmax")
    correct = tf.equal(label_argmax, preds, name = "correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name = "accuracy")
    
    # binary cross-entropy loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels = labels, logits = logits, name = "softmax_cross_entropy")
    loss = tf.reduce_mean(cross_entropy, name = "loss")

    # optimiser
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    return input_data, labels, optimizer, accuracy, loss, dropout_on, dropout_off
