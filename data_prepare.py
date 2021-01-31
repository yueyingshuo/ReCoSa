import tensorflow as tf
import csv
from hyperparams import Hyperparams as hp
import numpy as np
import os
import random
import codecs
def load_de_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab():
    vocab = [line.split()[0] for line in codecs.open('preprocessed/train.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


x_list, y_list, Sources, Targets = [], [], [], []

def create_dataset():
    #de2idx, idx2de = load_de_vocab()
    en2idx, idx2en = load_en_vocab()
    #file=os.path.join(fpath,fname)
    with open(hp.train_set, "r", encoding="utf-8") as read:
        csv_read = csv.reader(read)
        for line in csv_read:
            fields = [s.replace('__eou__', '.').replace('__eot__', '\n').strip() for s in line]
            context = fields[0].split('\n')
            x = []
            for s in context:
                if len(s.split()) == 0:
                    continue
                x.append([en2idx.get(word,1) for word in s.split()])
            response = fields[1]
            y=[en2idx.get(word,1) for word in response.split()]
            cands = None
            if len(fields) > 3:
                cands = [fields[i] for i in range(2, len(fields))]
                cands.append(response)
                random.shuffle(cands)
            if max(len(x),len(y))<=hp.maxlen:
                x_list.append(np.array(x))
                y_list.append(np.array(y))
                Sources.append(fields[0])
                Targets.append(fields[1])
            # Pad
        X = np.zeros([len(x_list), hp.max_turn, hp.maxlen], np.int32)
        Y = np.zeros([len(y_list), hp.maxlen], np.int32)
        X_length = np.zeros([len(x_list), hp.max_turn], np.int32)
        for i, (x, y) in enumerate(zip(x_list, y_list)):
            for j in range(len(x)):
                if j >= hp.max_turn:
                    break
                if len(x[j]) < hp.maxlen:
                    X[i][j] = np.lib.pad(x[j], [0, hp.maxlen - len(x[j])], 'constant', constant_values=(0, 0))
                else:
                    X[i][j] = x[j][:hp.maxlen]
                X_length[i][j] = len(x[j])
            # X[i] = X[i][:len(x)]
            # X_length[i] = X_length[i][:len(x)]
            Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))

    return X, X_length, Y, Sources, Targets




if __name__ == '__main__':

    X, X_length, Y, Sources, Targets=create_dataset()
   # print(X, X_length, Y, Sources, Targets)




