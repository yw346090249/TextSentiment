# -*- coding:utf-8 -*-
import numpy as np
import codecs
import time
import tools

# load sentiment data
# Input: @path of the sentiment data
# Output:@merge the class2 to class1
def load_class2_to_class1(class_transition_path):
    file = codecs.open(class_transition_path, "r", encoding='utf-8', errors='ignore')
    class2_class1 = {}
    for line in file.readlines():
        line = line.strip()
        terms = line.split('\t')
        class2_class1[terms[0]] = terms[1]
    return class2_class1

# load class2 data
# Input: @path of the class2 label
#        @path of the class2 data
# Output:@sample list
#        @label list
#        @ID2label transition table
def load_training_data_class2(label_path, data_path):
    classname2ID = {}
    ID2label = {}
    file = codecs.open(label_path, "r", encoding='utf-8', errors='ignore')
    count = 0
    for line in file.readlines():
        classname2ID[line.strip()] = count
        ID2label[count] = line.strip()
        count += 1

    samples = []
    labels = []
    file = codecs.open(data_path, "r", encoding='utf-8', errors='ignore')
    for line in file.readlines():
        line = line.strip()
        terms = line.split('\t')
        samples.append(terms[1])
        labels.append(classname2ID[terms[0]])
    return samples, labels, ID2label

# load sentiment data
# Output:@accuracy of the model class2
#        @accuracy of the model class1
def train_class():
    samples, labels, ID2label = load_training_data_class2(tools.PATH+'/data/class2_labels.txt', tools.PATH + '/data/manually_labeled_data_class2.txt')  #load class data
    dict = tools.build_dict(samples, tools.MAX_NB_WORDS)    #bulid dict
    print(len(dict))
    tools.save_dict(dict)    #save the dict to local
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
    training_ratio = 0.9    #setting the training data percentage
    N_train = int(NUM * training_ratio)
    X_train = X[:N_train]
    y_train = y[:N_train]
    X_val = X[N_train:]
    y_val = y[N_train:]
    samples_val = samples[N_train:]
    labels_val = labels[N_train:]
    sample_weights = np.ones(len(y_train))  #initialize the sample weight as all 1

    model = tools.define_model(tools.MAX_SEQUENCE_LENGTH, embedding_matrix, nb_words, EMBEDDING_DIM, N_label)
    model_save_path = 'code\model_class2'   #save the best model
    model = tools.train_model(model, X_train, y_train, X_val, y_val, sample_weights, model_save_path)

    score, accuracy_class2 = model.evaluate(X_val, y_val, batch_size=2000)   #get the score and acc for the model
    print('Test score:', score)
    print('Test accuracy:', accuracy_class2)

    pred = model.predict(X_val, batch_size=2000)    #get the concrete predicted value for each text
    labels_pred = tools.probs2label(pred)   #change the predicted value to labels

    #save the wrong result for class2
    writer_class2 = codecs.open(tools.PATH+'/data/wrong_analysis/class2_wrong_result.txt', "w", encoding='utf-8', errors='ignore')
    for i in range(len(labels_val)):
        if labels_val[i]!=labels_pred[i]:
            writer_class2.write(samples_val[i] +'\t'+ ID2label[labels_val[i]] +'\t'+ ID2label[labels_pred[i]] + '\n')
    writer_class2.flush()
    writer_class2.close()

    class2_class1 = load_class2_to_class1(tools.PATH+'/data/class2_class1.txt') #merge the class2 to class1
    N_class1_true = 0
    worng_class = []
    for i in range(len(labels_val)):
        if class2_class1[ID2label[labels_val[i]]]==class2_class1[ID2label[labels_pred[i]]]:
            N_class1_true += 1
        else:
            worng_class.append(class2_class1[ID2label[labels_val[i]]]+"\t"+class2_class1[ID2label[labels_pred[i]]]+"\t"+samples_val[i])

    #save the wrong result for class1
    writer = codecs.open(tools.PATH+'/data/wrong_analysis/class1_wrong_result.txt', "w", encoding='utf-8', errors='ignore')
    writer.write("original_label"+"\t"+"predict_label"+"\t"+"sample"+"\n")
    for item in worng_class:
        writer.write(item + '\n')
    writer.flush()
    writer.close()

    accuracy_class1 = N_class1_true/len(labels_val)
    print(accuracy_class1)
    return accuracy_class2, accuracy_class1

if __name__ == '__main__':
    start = time.clock()
    train_class()
    print(time.clock() - start) #show the running time of the training process