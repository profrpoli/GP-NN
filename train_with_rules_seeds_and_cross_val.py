from scipy import stats
from functools import partial
from multiprocessing import Pool
from operator import itemgetter
import numpy as np
import pylab as P
import pandas as pd
import time
import pathlib
import scipy as S
import os
from sklearn.model_selection import train_test_split


def sigmoid(x):
    x[(x / 700) > 1] = 700
    x[(x / 700) < -1] = -700
    x = 1 / (1 + np.exp(-x))
    return x


def sigmoid_grad(x):
    return (x) * (1 - x)


def tanh(x):
    x[(x / 700) > 1] = 700
    x[(x / 700) < -1] = -700
    return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))


def tanh_grad(x):
    return (1 - x ** 2)


def relu(x):
    return np.maximum(0, x)


def calc_rule(x, o, d, e, rule):
    if rule == 'sbp':
        return ((x.T).dot(d * e))
    elif rule== 'nlr':
        return ((x * x).T.dot(d * d * e * e) + (x).T.dot(o * d * e) + (3 * x * x).T.dot(d * e))

def limitation(w):
    np.nan_to_num(w)
    w[w > wg_limit] = wg_limit
    w[w < -wg_limit] = -wg_limit
    return (w)


class NeuralNetwork:
    def __init__(self, filename, epoch, lr, mu, inp_dim, target_dim, hidden, test_split, seed, activation,
                 rule, batch_size=30, data_norm=True, shuffle=True):  # Shape of layers
        np.random.seed(seed)
        ####Reading data from csv file

        self.structure = []
        self.structure.append(1 + inp_dim)
        for h in hidden:
            self.structure.append(1 + h)

        self.W = []
        self.delW = []
        self.prevdelW = []
        for i in range(len(self.structure) - 1):
            self.W.append(0.1 * np.random.randn(self.structure[i], self.structure[i + 1] - 1))
            self.delW.append(np.zeros((self.structure[i], self.structure[i + 1] - 1)))
            self.prevdelW.append(np.zeros((self.structure[i], self.structure[i + 1] - 1)))
        self.W.append(0.1 * np.random.randn(self.structure[-1], target_dim))
        self.delW.append(np.zeros((self.structure[-1], target_dim)))
        self.prevdelW.append(np.zeros((self.structure[-1], target_dim)))

        self.filename = filename
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.activation = activation
        self.rule=rule
        self.epoch = epoch
        self.inp_dim = inp_dim
        self.target_dim = target_dim
        self.lr = lr
        self.mu = mu
        self.inp_dim = inp_dim
        self.target_dim = target_dim
        self.er_layer = [None] * (len(self.structure))
        self.dr_layer = [None] * (len(self.structure))

    def upload_data(self, filename):
        realdata = np.array(pd.read_csv(mypath + '/data/' + filename+'.csv', delimiter=","), dtype=float)
        realdata = self.enumerate_classes(realdata)
        X=realdata[:, :inp_dim]
        y=realdata[:, inp_dim:]
        X, self.test_X, y, self.test_Y = train_test_split(X, y, test_size = test_split, random_state = 42)

        self.folds_X = [None] * cv_fold_numb
        self.folds_Y = [None] * cv_fold_numb
        for i in range(cv_fold_numb-1):
            X, self.folds_X[i], y, self.folds_Y[i] = train_test_split(X, y, test_size=1/(cv_fold_numb-i), random_state=42)
        self.folds_X[-1] = X
        self.folds_Y[-1] = y

        self.test_Y = np.array(self.test_Y, dtype=int).flatten()
        if data_norm:
            self.test_X = (self.test_X - self.test_X.min(axis=0)) / (self.test_X.max(axis=0) - self.test_X.min(axis=0))

    def enumerate_classes(self, a):
        a = np.array(sorted(a, key=itemgetter(len(a[0]) - 1)))
        a[:, -1] = a[:, -1] - np.min(a[:, -1])
        c = a[:, -1]
        b = np.array(list(set(a[:, -1])))
        d = np.arange(len(b))
        for i in range(len(b)):
            c[c == b[i]] = d[i]
        a[:, -1] = c
        return (a)

    def fold_concantenate(self, valid_num, temp_folds_X, temp_folds_Y):
        if cv_fold_numb > 1:
            self.valid_X = temp_folds_X[valid_num]
            self.valid_Y = temp_folds_Y[valid_num]
            del temp_folds_X[valid_num]
            del temp_folds_Y[valid_num]
            self.train_X = np.concatenate((temp_folds_X))
            self.train_Y = np.concatenate((temp_folds_Y))
        else:
            self.valid_X = self.test_X
            self.valid_Y = self.test_Y
            self.train_X = temp_folds_X[0]
            self.train_Y = temp_folds_Y[0]

        self.sample_numb = len(self.train_X)

        self.train_Y = np.array(self.train_Y, dtype=int).flatten()
        self.valid_Y = np.array(self.valid_Y, dtype=int).flatten()

        if data_norm:
            self.train_X = (self.train_X - self.train_X.min(axis=0)) / (self.train_X.max(axis=0) - self.train_X.min(axis=0))
            self.valid_X = (self.valid_X - self.valid_X.min(axis=0)) / (self.valid_X.max(axis=0) - self.valid_X.min(axis=0))


    def get_data(self, batch):
        if self.shuffle:
            perm = np.random.permutation(len(self.train_X))
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        for i in range(batch):
            self.mini_X = [self.train_X[j:j + batch] for j in range(0, len(self.train_X), batch)]
            self.mini_Y = [self.train_Y[j:j + batch] for j in range(0, len(self.train_Y), batch)]

    # forward propagation
    def forward_propagation(self, x, y):
        self.softmax = np.empty(self.target_dim)
        self.dr_out = np.empty(self.target_dim)
        self.layers = []
        self.layers.append(np.insert(x, 0, 1., axis=1))  # input data and insert biases

        if self.activation == 'sigmoid':
            for i in range(len(self.W) - 1):  # calculate hidden layers and insert biases
                self.layers.append(np.insert(sigmoid(np.dot(self.layers[i], self.W[i])), 0, 1., axis=1))

        elif self.activation == 'tanh':
            for i in range(len(self.W) - 1):
                self.layers.append(tanh(np.dot(self.layers[i], self.W[i])))
                self.layers[-1] = np.insert(self.layers[-1], 0, 1., axis=1)

        elif self.activation == 'relu':
            for i in range(len(self.W) - 1):  # calculate hidden layers and insert biases
                self.layers.append(np.insert(relu(np.dot(self.layers[i], self.W[i])), 0, 1., axis=1))

        # softmax layer
        self.out = np.dot(self.layers[-1], self.W[-1])
        self.out[self.out > 300] = 300
        self.out[self.out < -300] = -300
        exp_out = np.exp(self.out)
        self.softmax = exp_out / np.sum(exp_out, axis=1, keepdims=True)

        # compute average cross-entropy loss
        corect_logsoftmax = -np.log(self.softmax[range(len(y)), y])
        data_loss = np.sum(corect_logsoftmax) / self.sample_numb  # / len(y)
        # reg_loss = 0.5 * reg * self.W.sum()**2
        loss = data_loss  # + reg_loss

        # gradient on softmax
        self.dr_out = self.softmax
        self.dr_out[range(len(y)), y] -= 1
        self.dr_out /= len(y)

        return (loss)

    # Backpropagation
    def backpropagation(self):
        self.delW[-1] = (self.layers[-1].T).dot(self.dr_out)
        self.delW[-1] = limitation(self.delW[-1])

        if self.activation == 'sigmoid':
            self.er_layer[-1] = self.dr_out.dot(self.W[-1][1:].T)
            self.dr_layer[-1] = sigmoid_grad(self.layers[-1][:, 1:])

            for i in range(len(self.layers) - 2, -1, -1):
                x = self.layers[i]
                d = self.dr_layer[i + 1]
                e = self.er_layer[i + 1]
                o = self.layers[i + 1][:, 1:]
                # w = self.W[i]
                self.delW[i] = calc_rule(x, o, d, e, self.rule)
                self.delW[i] = limitation(self.delW[i])

                self.er_layer[i] = (self.er_layer[i + 1] * self.dr_layer[i + 1]).dot(self.W[i][1:].T)
                self.dr_layer[i] = sigmoid_grad(self.layers[i][:, 1:])

        elif self.activation == 'tanh':
            self.er_layer[-1] = self.dr_out.dot(self.W[-1][1:].T)
            self.dr_layer[-1] = tanh_grad(self.layers[-1][:, 1:])

            for i in range(len(self.layers) - 2, -1, -1):
                x = self.layers[i]
                d = self.dr_layer[i + 1]
                e = self.er_layer[i + 1]
                o = self.layers[i + 1][:, 1:]
                # w = self.W[i]
                self.delW[i] = calc_rule(x, o, d, e, self.rule)
                self.delW[i] = limitation(self.delW[i])

                self.er_layer[i] = (self.er_layer[i + 1] * self.dr_layer[i + 1]).dot(self.W[i][1:].T)
                self.dr_layer[i] = tanh_grad(self.layers[i][:, 1:])


        elif self.activation == 'relu':
            self.dr_layer[-1] = np.ones(np.array(self.layers[-1][:, 1:]).shape)
            self.dr_layer[-1][np.array(self.layers[-1][:, 1:]) <= 0] = 0
            self.er_layer[-1] = self.dr_out.dot(self.W[-1][1:].T)

            for i in range(len(self.er_layer) - 2, -1, -1):
                x = self.layers[i]
                d = self.dr_layer[i + 1]
                e = self.er_layer[i + 1]
                o = self.layers[i + 1][:, 1:]
                # w = self.W[i]
                self.delW[i] = calc_rule(x, o, d, e, self.rule)
                self.delW[i] = limitation(self.delW[i])

                self.dr_layer[i] = np.ones(np.array(self.layers[i][:, 1:]).shape)
                self.dr_layer[i][np.array(self.layers[i][:, 1:]) <= 0] = 0
                self.er_layer[i] = (self.er_layer[i + 1] * self.dr_layer[i + 1]).dot(self.W[i][1:].T)

        # self.delW[0] = (x.T).dot(self.dr_hidden[0] * self.er_hidden[0])
        # self.delb[0] = np.sum(self.dr_hidden[0] * self.er_hidden[0])

    def update_weight(self):
        for i in range (len(self.W)):
            self.W[i] += -(self.delW[i] * self.lr + self.prevdelW[i] * self.mu)
            self.prevdelW[i] = self.delW[i] * self.lr + self.prevdelW[i] * self.mu

    def test(self):
        ls = self.forward_propagation(self.train_X, self.train_Y)
        predicted_class = np.argmax(self.out, axis=1)
        train_ac = np.mean(predicted_class == self.train_Y)

        ls = self.forward_propagation(self.valid_X, self.valid_Y)
        predicted_class = np.argmax(self.out, axis=1)
        valid_ac = np.mean(predicted_class == self.valid_Y)

        return (train_ac, valid_ac)


def evaluate(w, n):
    lossarray = np.zeros(n.epoch)
    train_ac = np.zeros(n.epoch)
    valid_ac = np.zeros(n.epoch)
    for i in range(cv_fold_numb):
        n.fold_concantenate(i, n.folds_X.copy(), n.folds_Y.copy())
        n.W = w.copy()
        for t in range(n.epoch):
            n.get_data(n.batch_size)
            #loss = 0
            for x, y in zip(n.mini_X, n.mini_Y):
                loss_mini = n.forward_propagation(x, y)
                #loss += loss_mini
                n.backpropagation()
                n.update_weight()
            loss = n.forward_propagation(np.concatenate(n.mini_X), np.concatenate(n.mini_Y))
            accuracy = n.test()
            train_ac[t] += accuracy[0] / cv_fold_numb
            valid_ac[t] += accuracy[1] / cv_fold_numb
            lossarray[t] += loss / cv_fold_numb
    return (lossarray, train_ac, valid_ac)


def param_save():
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    file = open(folder + '/parameters.txt', "a")
    file.writelines('rule : %s \nproblem: %s \t epochs: %d\tstructure: %d-%s-%d '
                    '\t batch_size: %d,\tdata norm: %s\n lr: %s\n\n'
                    % (ruleexp, filename, epoch, inp_dim, str(hidden),
                       target_dim, batch_size, str(data_norm), str(lr)))
    file.close()


def csv_save(bp, lr, graph_type):
    np.savetxt('%s/learning rate/%.2f/%s_%s.csv' % (folder, lr, n.filename, graph_type), bp, delimiter=",")

filename='iris'
number_of_seeds = 20
epoch = 1000
lr = 0.8
inp_dim = 4
target_dim = 3
hidden = [15, 15]
batch_size = 120
test_split = 0.2
activation = 'sigmoid'
cv_fold_numb = 1
data_norm = True
shuffle = False
wg_limit = 1e30
mypath = str(os.path.dirname(__file__))

ruleexps = ['s',
            's*(s+o+3*x)',
            ]
rules = [ 'sbp',
          'nlr',
         ]  # should be 'sbp' or 'nlr' but this can be increased the if there are more rules to be compared

lr_bp_train_ac = []
lr_bp_test_ac = []
lr_bp_train_ac_trm = []
lr_bp_test_ac_trm = []

if __name__ == '__main__':
    pool = Pool(20)
    for rule, ruleexp in zip(rules, ruleexps):
        folder1 = ('%s/testing_rules/csv file with cross val_size%d/%s/%s' % (mypath, cv_fold_numb, activation, rule))
        folder = ('%s/%s' % (folder1, filename))
        param_save()
        w_list = []
        for seed in range(number_of_seeds):
            n = NeuralNetwork(filename=filename, epoch=epoch, lr=lr, mu=0, inp_dim=inp_dim, target_dim=target_dim,
                              hidden=hidden, batch_size=batch_size, test_split=test_split, activation=activation, rule=rule,
                              seed=seed, data_norm=data_norm, shuffle=shuffle)
            n.upload_data(filename)

            w_list.append(n.W)

        print('rule:',rule, 'learningrate=', lr)
        pathlib.Path('%s/learning rate/%.2f' % (folder, lr)).mkdir(parents=True, exist_ok=True)

        n.lr = lr
        begin = time.time()
        bp_results = np.array(pool.map(partial(evaluate, n=n), w_list))
        tss_bp = bp_results[:, 0]
        train_ac_bp = bp_results[:, 1]
        valid_ac_bp = bp_results[:, 2]

        print('%s (%s) has been computed in \t%s seconds:' % (n.filename, rule, str(time.time() - begin)))

        csv_save(train_ac_bp, lr, 'train_ac')
        csv_save(valid_ac_bp, lr, 'valid_ac')
        csv_save(tss_bp, lr, 'loss')
