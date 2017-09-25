#!/user/bin/python2
# -*- coding: utf-8 -*-
import random
import math


class ml_base(object):
    def __init__(self, dataset, test_rate):
        (self.test_data, self.test_label,
         self.train_data, self.train_label) = get_data(dataset=dataset,
                                                       test_rate=test_rate)

    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.00000000001


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
    elif dataset == 'car':
        label_map = {'unacc': 0, 'acc': 1, 'good': 1, 'vgood': 1}
        buy_map = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
        maint_map = {'vhigh': 3, 'high': 2, 'med': 1, 'low': 0}
        doors_map = {'2': 0, '3': 1, '4': 2, '5more': 3}
        persons_map = {'2': 0, '4': 1, 'more': 2}
        lug_map = {'small': 0, 'med': 1, 'big': 2}
        safe_map = {'low': 0, 'med': 1, 'high': 2}
        maps = [buy_map, maint_map, doors_map, persons_map, lug_map, safe_map]
        with open('./dataset/car.data', 'r') as f:
            line = f.readline()
            while line:
                row = line[:-1].split(',')
                temp = []
                label.append(label_map[row[-1]])
                for (i, m) in enumerate(maps):
                    temp.append(m[row[i]])
                data.append(temp)
                line = f.readline()
    zip_data = zip(data, label)
    random.shuffle(zip_data)
    data = [i[0] for i in zip_data]
    label = [i[1] for i in zip_data]
    len_test = int(test_rate * len(data))
    return (data[:len_test], label[:len_test],
            data[len_test:], label[len_test:])
