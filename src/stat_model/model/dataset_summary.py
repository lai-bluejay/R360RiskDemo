# -*- coding:utf-8 -*-
import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../.." % root)
sys.path.append("%s/.." % root)
sys.path.append("%s/../../.." % root)
sys.path.append("{0:s}".format(root))

__author__ = 'Charles_Lai'
__author_email__ = 'lai.bluejay@gmail.com'

import random
from utils import exe_time
from sklearn.cross_validation import train_test_split


class DatasetSummary(object):
    def __init__(self):
        pass

    def get_random_train_validation_old(self, feature, label):
        length = len(feature)
        random_list = []
        for i in range(length):
            random_list.append(i)
        random.shuffle(random_list)
        random_feature = []
        random_label = []
        for j in random_list:
            random_feature.append(feature[j])
            random_label.append(label[j])
        return random_feature[:int(0.88 * len(feature))], random_label[:int(0.88 * len(feature))], random_feature[int(
                0.88 * len(feature)):], random_label[int(0.88 * len(feature)):]

    def get_random_train_validation_set(self, sample_space, random_state=31, train_size=0.9):
        """
        返回训练集和测试集
        :param sample_space: 样本空间
        :param random_state: 随机状态，一般为质数比较合理
        :param train_size: 训练集的大小
        :return:
        """
        train_set, vali_set = train_test_split(sample_space, random_state=31, train_size=0.9)
        return train_set, vali_set

    @exe_time
    def get_dataset_segments(self, feature, label):

        length = len(feature)
        train_pos = int(length * 0.8)
        vali_pos = int(length * 0.9)

        x_train = feature[:train_pos]
        y_train = label[:train_pos]

        x_validation = feature[train_pos:vali_pos]
        y_validation = label[train_pos:vali_pos]

        x_test = feature[vali_pos:length]
        y_test = label[vali_pos:length]

        print("总样本数:" + str(length))
        return x_train, y_train, x_validation, y_validation, x_test, y_test

    @exe_time
    def get_train_test_dataset_segments(self, feature, label):

        length = len(feature)
        train_pos = int(length * 0.9)
        x_train = feature[:train_pos]
        y_train = label[:train_pos]
        x_test = feature[train_pos:length]
        y_test = label[train_pos:length]

        print("总样本数:" + str(length))
        return x_train, y_train, x_test, y_test

    def get_random_train_vali_segments(self, feature, label):

        length = len(feature)
        train_pos = int(length * 0.8)
        test_pos = int(length * 0.9)

        x_train_vali = feature[:test_pos]
        y_train_vali = label[:test_pos]

        x_train, y_train, x_validation, y_validation = self.get_random_train_validation(x_train_vali, y_train_vali)

        x_test = feature[test_pos:length]
        y_test = label[test_pos:length]

        print("总样本数:" + str(length))
        return x_train, y_train, x_validation, y_validation, x_test, y_test

    @exe_time
    def print_label_dataset_detail(self, y_train, y_validation, y_test):
        output_string = ""
        with open(root + '/../feature/data/dataset_description.txt', 'a') as fo:
            output_string = "train_set:" + self.print_one_label_dataset_detail(y_train)+"validation_set:" \
                            +self.print_one_label_dataset_detail(y_validation) +  "test_set:" + \
                            self.print_one_label_dataset_detail(y_test)
            fo.write(output_string)
        return output_string

    def print_one_label_dataset_detail(self, y_train):
        train_length = len(y_train)
        train_label_1 = sum(y_train)
        tmp_string = "样本数:" + str(train_length) + "\t" + "标签为1样本数:" + str(
                train_label_1) + "\t" + "标签为1样本数占比:" + str(
                round(train_label_1 / float(train_length), 3)) + "\n"
        return tmp_string
