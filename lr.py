#!/user/bin/python2
# -*- coding: utf-8 -*-
import random
import math
from ml_base import ml_base
from evaluator import evaluator


class logistic_regressor(ml_base):
    def __init__(self, dataset, test_rate, train_round=1000,
                 normalization=None, batch_size=32, learning_rate=0.05):
        ml_base.__init__(self, dataset, test_rate)
        self.train_round = train_round
        self.normalization = normalization
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.feature_num = len(self.train_data[0])
        self.w = [random.random() for i in range(self.feature_num)]

    def get_class(self, x, threshold=0.5):
        if x > threshold:
            return 1
        else:
            return 0

    def get_loss(self):
        loss = 0
        for i in xrange(len(self.train_data)):
            z = self.sigmoid(
                sum(map(lambda a, b: a * b, self.w, self.train_data[i])))
            loss += self.train_label[i] * math.log(z) + (
                1 - self.train_label[i]) * math.log(1 - z + 0.00000001)
        return loss / len(self.train_data)

    def train_gradiand_descend(self):
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

    def get_test(self):
        pred = []
        for i in xrange(len(self.test_data)):
            temp = 0
            for j in xrange(self.feature_num):
                temp += self.w[j] * self.test_data[i][j]
            pred.append(self.get_class(self.sigmoid(temp)))
        return pred


if __name__ == '__main__':
    ml = logistic_regressor('iris', 0.2)
    ml.train_gradiand_descend()
    pred = ml.get_test()
    print pred, ml.test_label
    print ml.train_label
    lr_eval = evaluator(ml.test_label, pred, 'logistic regression')
