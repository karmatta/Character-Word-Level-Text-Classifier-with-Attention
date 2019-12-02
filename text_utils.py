# Title: Set of utility functions for text processing
# Author: Karthik D
# ----------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter


PAD = '<PAD>'


# Func to make vocab from df column
def make_vocab(tokens):
    vocab = set()
    if type(tokens) == str:
        vocab.update(tokens.split())
    else:
        tokens.str.split().apply(vocab.update)
    while '' in vocab:
        vocab.remove('')
    vocab.add(PAD)
    print("Vocab size:", len(vocab))
    return vocab


# Make word to index dict
def make_w2i(vocab):
    w2i = {}
    i=0
    for word in vocab:
        if not word in w2i:
            w2i[word] = i
            i+=1
    return w2i


# Wrapper function to make a torch variable
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# Function to make word padding
def padding(data, max_sent_len, pad_token):
    pad_len = max(0, max_sent_len - len(data))
    data += [pad_token] * pad_len
    data = data[:max_sent_len]
    return data


# Wrapper to make a tensor from a list/np
def make_tensor_torch(data, w2i, max_len):
    #seq_len = max(data.str.split().apply(len))
    ret_data = [padding([w2i[word] for word in row.split()], max_len, w2i[PAD]) for row in data]
    return to_var(torch.LongTensor(ret_data))


# Wrapper to make a tensor from a list/np
def make_tensor_np(data, w2i, max_len):
    #seq_len = max(data.str.split().apply(len))
    ret_data = [padding([w2i[word] for word in row.split()], max_len, w2i[PAD]) for row in data]
    return np.array(ret_data)


# one hot encode sequence
def one_hot_encode(sequence):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(len(vocab))]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)


# a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# Get pairs of predictors and target
def get_pair(n_in, n_out=None):
    X = n_in
    # reshape as 3D
    X = X.reshape((1, X.shape[0]))
    if n_out is not None:
        y = n_out[0]
        # reshape as 3D
        y = y.reshape((1, y.shape[0]))
    else:
        y = None
    return X,y


#Get a list of keys from dictionary which has value that matches with any value in given list of values
def getKeysByValues(listOfValues):
    listOfKeys = list()
    listOfItems = w2i.items()
    for item  in listOfItems:
        if item[1] in listOfValues and item[0] != PAD:
            listOfKeys.append(item[0])      
    return  listOfKeys


# Func to get unique characters and uncommon ones
def get_characters(text_column, THRESHOLD):
    # Get the # of unique symbols
    unique_symbols = Counter()

    for _, message in text_column.iteritems():
        unique_symbols.update(message)
    print("Unique symbols:", len(unique_symbols))
    
    # Find symbols that appear fewer times than the threshold:
    uncommon_symbols = list()
    
    # Get uncommon symbols
    for symbol, count in unique_symbols.items():
        if count < THRESHOLD:
            uncommon_symbols.append(symbol)
            
    return unique_symbols, uncommon_symbols
