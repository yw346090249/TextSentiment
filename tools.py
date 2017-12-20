# -*- coding:utf-8 -*-
import numpy as np
import codecs
import operator
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional, Input, TimeDistributed, Convolution1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras import backend as K
from keras import layers

# External Path for the whole Project
PATH = 'D:/MSRA/surface_comments_analysis'
# constant values
MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 10000

# Build the dict for the input text
# Input: @original text
#        @Max number of words: 10000
# Output:@the dict of the original text
def build_dict(texts, MAX_NB_WORDS):
    char2num = {}
    for text in texts:
        text = ''.join(text.split('\t'))
        for char in text:
            char2num[char] = char2num.setdefault(char, 0) + 1
    char2num_sorted = sorted(char2num.items(), key=operator.itemgetter(1), reverse=True)
    dict = {'PADDING':0,'UNK':1}    #Define PADDING as 0 and UNKNOWN as 1
    for i in range(len(char2num_sorted)):
        if i > MAX_NB_WORDS:
            break
        item = char2num_sorted[i]
        dict[item[0]] = len(dict)   #For new words the dict numbel equals to the dict length
    return dict

# save dict to local
# Input: @dict produced by build_dict function
def save_dict(dict):
    writer_dict = codecs.open(PATH+'/code/dict.txt', "w", encoding='utf-8', errors='ignore')
    for each in dict:
        writer_dict.write(each+"\n")
    writer_dict.flush()
    writer_dict.close()

# change prob to predicted labels
# Input: @dict of the text
# Output:@embedding matrix of the dict
#        @Max number of words: 10000
#        @embedding dimension
def load_embedding(dict):
    embdict = {}
    num = 0
    import _pickle as pickle
    with open(PATH+'/code/segment/chinese-char500.bin', 'rb')as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        EMBEDDING_DIM = layer1_size
        print(vocab_size, layer1_size)
        while True:
            ch = f.readline()
            if not ch:
                break
            k = ch.find(b' ')
            word = ch[0:k].decode('utf-8')
            emb = ch[k + 1:-2].split()
            if word in dict.keys():
                for i in range(len(emb)):
                    emb[i] = float(emb[i])
                embdict[word] = emb
            num += 1

    print(len(embdict))
    print(len(dict))
    words = dict.keys()
    nb_words = len(dict)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word in words:
        if word in embdict:
            embedding_matrix[dict[word]] = embdict[word]
    return embedding_matrix, nb_words, EMBEDDING_DIM

# Change the text to sequence
# Input: @original text
#        @the dict of the original text
# Output:@the sequence type of the text
def text_to_sequence(text, dict):
    seq = []
    for char in text:
        if char in dict:
            seq.append(dict[char])
        else:
            seq.append(dict['UNK'])
    return seq

# Normalize the training data
# Input: @original text
#        @labels list
#        @number of labels in labels list
#        @the dict of the original text
#        @MAX sequence length:100
# Output:@array X for text; array y for labels
def normalize_training_data(texts, labels, N_label, dict, MAX_SEQUENCE_LENGTH):
    sequences = []
    for text in texts:
        seq = text_to_sequence(text, dict)
        sequences.append(seq)
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    X = np.asarray(X)
    y = []
    for label in labels:
        vec = np.zeros(N_label)
        vec[label] = 1
        y.append(vec)
    y = np.asarray(y)
    return X, y

# Only used for Model 3: add attention into Model2
def att_process(candidates, att, activation='tanh'):
    att_dim = K.int_shape(att)[-1]
    candidates2 = layers.TimeDistributed(layers.Dense(att_dim, activation=activation))(candidates)
    dotted = layers.dot([candidates2, att], axes=(2, 1), normalize=True)
    weights = layers.Activation('softmax')(dotted)
    weighted = layers.dot([candidates, weights], axes=(1, 1))
    return weighted, weights

# 3 models in this part, choose 1 for training
def define_model(MAX_SEQUENCE_LENGTH, embedding_matrix, nb_words, EMBEDDING_DIM, N_label):
    # Model 1 based on CNN
    # nb_filter = 500
    # filter_length = 3
    # hidden_dim = 100
    # model = Sequential()
    # model.add(Embedding(nb_words, EMBEDDING_DIM,  weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True))
    # model.add(Convolution1D(nb_filter=nb_filter, filter_length=filter_length,  border_mode='same', activation='relu'))
    # model.add(GlobalMaxPooling1D())
    # model.add(Dense(hidden_dim, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(N_label, activation='sigmoid'))
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


    # Model 2 based on Bi-lstm
    sequence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, mask_zero=True, trainable=True)(sequence) #
    blstm = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2), merge_mode='concat')(embedded)
    output = Dense(N_label, activation='softmax')(blstm)
    model = Model(input=sequence, output=output)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # Model 3 based on Attention+Bi-lstm
    # sequence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # embedded = Embedding(nb_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH,
    #                      mask_zero=True, trainable=True)(sequence)  #
    # blstm = Bidirectional(LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode='concat')(
    #     embedded)
    # lstm = LSTM(100, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(embedded)
    # blstm, weights = att_process(blstm, lstm)
    # output = Dense(N_label, activation='softmax')(blstm)
    # model = Model(input=sequence, output=output)
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

# model training process
# Input: @defined model
#        @training text(80%)
#        @training labels(80%)
#        @validation text(20%)
#        @validation labels(20%)
#        @sample weight: initial all 1
#        @saved model path
# Output:@array X for text; array y for labels
def train_model(model, X_train, y_train, X_val, y_val, sample_weights, path):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = path+'.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)
    model.fit(X_train, y_train, sample_weight=sample_weights, batch_size=50, epochs=5, shuffle=True, validation_data=(X_val, y_val),
              callbacks=[early_stopping, model_checkpoint])
    return model

# change prob to predicted labels
# Input: @prob list of text
# Output:@predicted labels list of text
def probs2label(pred):
    labels = []
    for probs in pred:
        probs = np.asarray(probs)
        index = np.argmax(probs)
        labels.append(index)
    return labels