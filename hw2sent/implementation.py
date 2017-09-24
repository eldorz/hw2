import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import tarfile
import re
import string

batch_size = 50
GLOVE_DIM = 50
GLOVE_MAX_VOCAB = 10000  # 400000 words in glove datasete
NUM_REVIEWS = 25000
WORDS_PER_REVIEW = 40

def preprocess(rawstring):
    # stopwords
    stops = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
        'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
        'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
        'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
        'about', 'against', 'between', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'only', 'own', 'same', 'so', 'than', 'too', 'can', 'will'}

    nobr = re.sub(r'<br>', ' ', rawstring)
    no_punct = ''.join(c for c in nobr if c not in string.punctuation)
    lower = no_punct.lower()
    words = lower.split()
    processed = []
    for w in words:
        if w not in stops: continue
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

    return input_data, labels, optimizer, accuracy, loss
