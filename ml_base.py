#!/user/bin/python2
# -*- coding: utf-8 -*-
import random
import math


def get_data(dataset='iris', test_rate=0.2):
    print 'Load %s data with %.2f test rate.' % (dataset, test_rate)
    data = []
    label = []
    if dataset == 'iris':
        label_map = {
            'Iris-setosa': 0,
            'Iris-versicolor': 1,
            'Iris-virginica': 1,
        }
        with open('./dataset/iris.data', 'r') as f:
            line = f.readline()
            while line:
                line = line[:-1]
                row = line.split(',')
                data.append(map(float, row[:-2]))
                label.append(label_map[row[-1]])
                line = f.readline()
            zip_data = zip(data, label)
            random.shuffle(zip_data)
            data = [i[0] for i in zip_data]
            label = [i[1] for i in zip_data]
    len_test = int(test_rate * len(data))
    return (data[:len_test], label[:len_test],
            data[len_test:], label[len_test:])


class ml_base(object):
    def __init__(self, dataset, test_rate):
        (self.test_data, self.test_label,
         self.train_data, self.train_label) = get_data(dataset=dataset,
                                                       test_rate=test_rate)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
