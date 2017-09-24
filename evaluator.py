#!/usr/bin/python
# -*- coding: utf-8 -*-

import math


class evaluator(object):
    """
    模型评估工具，计算：RMSE，正确率，confusion matrix，准确率，召回率，F1值
    get_weight(): 返回这个在计算score的权重（召回率：TN/(TN+FN)）
    """

    def __init__(self, label, pred, name):
        self.name = name
        self.rmse = math.sqrt(sum(
            (pred[i] - label[i]) ** 2 for i in xrange(len(pred))
        ) / float(len(pred)))
        pred = map(lambda x: int(x > 0.45), pred)
        self.correct_rate = sum(map(lambda x, y: int(x) == int(y),
                                    pred, label)) / float(len(pred))
        (self.tp, self.fp,
         self.tn, self.fn) = self.__get_confusion_matrix(label, pred)
        self.p = self.tp / (self.tp + self.fp + 0.01)  # 查准率/准确率
        self.r = self.tp / (self.tp + self.fn + 0.01)  # 查全率/召回率
        self.f1 = 2 * self.p * self.r / (self.p + self.r + 0.01)
        self.__print_eval()

    def __print_eval(self):
        print '\n' + '*' * 79
        print 'Evaluator for %s' % self.name
        print '  Root Mean Square Error = %.5f' % self.rmse
        print '  Correct rate   = %.5f' % self.correct_rate
        print '  True Positive  = %d' % self.tp
        print '  True Negative  = %d' % self.tn
        print '  False Positive = %d' % self.fp
        print '  False Negative = %d' % self.fn
        print '  Recall    = %.5f' % self.r
        print '  Precision = %.5f' % self.p
        print '  F1 = %.5f' % self.f1
        print '*' * 79 + '\n'

    def get_weight(self):
        return self.tn / (self.tn + self.fp)

    def __get_confusion_matrix(self, label, pred):
        self.tp = self.fp = self.tn = self.fn = 0.
        length = len(pred)
        for l, p in zip(label, pred):
            l = int(l)
            p = int(p)
            if l == p and l == 0:
                self.tn += 1
            elif l == p and l == 1:
                self.tp += 1
            elif l == 0 and p == 1:
                self.fp += 1
            elif l == 1 and p == 0:
                self.fn += 1
        return (self.tp, self.fp, self.tn, self.fn)
