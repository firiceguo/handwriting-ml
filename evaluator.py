#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt


class evaluator(object):
    """
    模型评估工具，计算：RMSE，正确率，confusion matrix，准确率，召回率，F1值, AUC
    """

    def __init__(self, label, pred, name):
        self.name = name
        self.rmse = math.sqrt(sum(
            (pred[i] - label[i]) ** 2 for i in xrange(len(pred))
        ) / float(len(pred)))
        self.pred = pred
        self.label = label
        pred = map(lambda x: int(x > 0.5), self.pred)
        self.correct_rate = sum(map(lambda x, y: int(x) == int(y),
                                    pred, label)) / float(len(pred))
        (self.tp, self.fp,
         self.tn, self.fn) = self.__get_confusion_matrix(label, pred)
        self.p = self.tp / (self.tp + self.fp + 0.00000001)  # 查准率/准确率
        self.r = self.tp / (self.tp + self.fn + 0.00000001)  # 查全率/召回率
        self.f1 = 2 * self.p * self.r / (self.p + self.r + 0.00000001)
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
        print '  F1  = %.5f' % self.f1
        print '  AUC = %.5f' % self.get_auc()
        print '*' * 79 + '\n'

    def __get_confusion_matrix(self, label, pred):
        tp = fp = tn = fn = 0.
        length = len(pred)
        for l, p in zip(label, pred):
            l = int(l)
            p = int(p)
            if l == p and l == 0:
                tn += 1
            elif l == p and l == 1:
                tp += 1
            elif l == 0 and p == 1:
                fp += 1
            elif l == 1 and p == 0:
                fn += 1
        return (tp, fp, tn, fn)

    def plot_roc(self, plot=True):
        x = [i / 1000. for i in xrange(1000)]
        fpr = []
        tpr = []
        for threshold in x:
            pred = map(lambda i: int(i > threshold), self.pred)
            tp, fp, tn, fn = self.__get_confusion_matrix(self.label, pred)
            fpr.append(fp / (fp + tn + 0.00000001))
            tpr.append(tp / (tp + fn + 0.00000001))
        if plot:
            plt.title('ROC')
            plt.xlabel('FPR = FP / (FP + TN)')
            plt.ylabel('TPR = TP / (TP + FN)')
            plt.plot(fpr, tpr)
            plt.show()
        else:
            return fpr, tpr

    def get_auc(self):
        fpr, tpr = self.plot_roc(plot=False)
        temp = sorted(zip(fpr, tpr), key=lambda a: a[0])
        fpr = [t[0] for t in temp]
        tpr = [t[1] for t in temp]
        auc = 0
        for i in xrange(1, len(fpr)):
            auc += (fpr[i] - fpr[i - 1]) * tpr[i]
        return auc
