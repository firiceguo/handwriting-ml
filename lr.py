#!/user/bin/python2
# -*- coding: utf-8 -*-
import random
import math
from ml_base import ml_base
from evaluator import evaluator


class logistic_regressor(ml_base):
    def __init__(self, dataset, test_rate, train_round=300,
                 normalization=None, batch_size=32, learning_rate=0.005):
        ml_base.__init__(self, dataset, test_rate)
        self.train_round = train_round
        self.normalization = normalization
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.feature_num = len(self.train_data[0])
        self.w = [random.random() for i in range(self.feature_num)]

    def get_loss(self):
        loss = 0
        for i in xrange(len(self.train_data)):
            z = self.sigmoid(
                sum(map(lambda a, b: a * b, self.w, self.train_data[i])))
            loss += self.train_label[i] * math.log(z) + (
                1 - self.train_label[i]) * math.log(1 - z + 0.00000001)
        return loss / len(self.train_data)

    def train_gd(self):
        for i in xrange(self.train_round):
            delta = [0 for ii in xrange(self.feature_num)]
            for j in xrange(len(self.train_data)):
                for k in xrange(self.feature_num):
                    z = self.sigmoid(
                        sum(map(lambda x, y: x * y,
                                self.w, self.train_data[j])))
                    delta[k] += (
                        z - self.train_label[j]) * self.train_data[j][k]
            for k in xrange(self.feature_num):
                self.w[k] -= delta[k] * self.learning_rate
            if i % 100 == 0:
                print 'Round %d, loss:%.5f' % (i, self.get_loss())

    def train_sgd(self):
        for i in xrange(self.train_round):
            data = self.train_data
            random.shuffle(data)
            delta = [0 for ii in xrange(self.feature_num)]
            for j in xrange(len(data)):
                for k in xrange(self.feature_num):
                    z = self.sigmoid(
                        sum(map(lambda x, y: x * y, self.w, data[j])))
                    delta[k] += (z - self.train_label[j]) * data[j][k]
                if j % self.batch_size == 0 and j != 0:
                    for k in xrange(self.feature_num):
                        self.w[k] -= delta[k] * self.learning_rate
                    delta = [0 for ii in xrange(self.feature_num)]
            if i % 100 == 0:
                print 'Round %d, loss:%.5f' % (i, self.get_loss())

    def get_pred(self):
        pred = []
        for i in xrange(len(self.test_data)):
            temp = 0
            for j in xrange(self.feature_num):
                temp += self.w[j] * self.test_data[i][j]
            pred.append(self.sigmoid(temp))
        return pred


if __name__ == '__main__':
    ml = logistic_regressor('car', 0.2)
    ml.train_gd()
    pred = ml.get_pred()
    lr_eval = evaluator(ml.test_label, pred, 'logistic regression')
    lr_eval.plot_roc()
