# -*- coding:utf-8 -*-
import numpy as np
import codecs
import time
import tools

# load sentiment data
# Input: @path of the sentiment data
# Output:@sample list
#        @label list
#        @ID2label transition table
def load_training_data_sentiment(path):
    samples = []
    labels = []
    file = codecs.open(path, "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        line = line.strip()
        terms = line.split('\t')
        samples.append(terms[1])
        labels.append(int(terms[0]))
    ID2label = {}
    ID2label[0] = '中性'
    ID2label[1] = '正面'
    ID2label[2] = '负面'
    return samples, labels, ID2label

# load sentiment data
# Output:@accuracy of the model
def train_sentiment():
    samples, labels, ID2label = load_training_data_sentiment(tools.PATH + '/data/manually_labeled_data_sentiment.txt')  #load data
    dict = tools.build_dict(samples, tools.MAX_NB_WORDS)    #bulid dict
    tools.save_dict(dict)   #save the dict to local
    print(len(dict))
    embedding_matrix, nb_words, EMBEDDING_DIM = tools.load_embedding(dict)  #load embedding
    N_label = len(ID2label)
    X, y = tools.normalize_training_data(samples, labels, N_label, dict, 100)   #normalize the input data
    print(len(X))
    print(len(y))

    NUM = len(X)
    indices = np.arange(NUM)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    samples = np.asarray(samples)
    samples = samples[indices]
    labels = np.asarray(labels)
    labels = labels[indices]
    training_ratio = 0.8    #setting the training data percentage
    N_train = int(NUM * training_ratio)
    X_train = X[:N_train]
    y_train = y[:N_train]
    X_val = X[N_train:]
    y_val = y[N_train:]
    samples_val = samples[N_train:]
    labels_val = labels[N_train:]
    sample_weights = np.ones(len(y_train))  #initialize the sample weight as all 1

    model = tools.define_model(tools.MAX_SEQUENCE_LENGTH, embedding_matrix, nb_words, EMBEDDING_DIM, N_label)
    model_save_path = 'code\model_sentiment' #save the best model
    model = tools.train_model(model, X_train, y_train, X_val, y_val, sample_weights, model_save_path)

    score, acc = model.evaluate(X_val, y_val, batch_size=2000)  #get the score and acc for the model

    print('Test score:', score)
    print('Test accuracy:', acc)

    pred = model.predict(X_val, batch_size=2000) #get the concrete predicted value for each text
    labels_pred = tools.probs2label(pred)      #change the predicted value to labels
    #save the wrong result
    writer_sentiment = codecs.open(tools.PATH+'/data/wrong_analysis/sentiment_wrong_result.txt', "w", encoding='utf-8', errors='ignore')
    for i in range(len(labels_val)):
        if labels_val[i]!=labels_pred[i]:
            writer_sentiment.write(samples_val[i] +'\t'+ ID2label[labels_val[i]] +'\t'+ ID2label[labels_pred[i]] + '\n')
    writer_sentiment.flush()
    writer_sentiment.close()
    return acc

if __name__ == '__main__':
    start = time.clock()
    train_sentiment()
    print(time.clock() - start) #show the running time of the training process