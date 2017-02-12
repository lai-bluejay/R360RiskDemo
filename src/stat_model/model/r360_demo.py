#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2017 yongqianbao.com inc. All rights reserved.
R360_risk.r360_demo was created on 11/02/2017.
Author: Charles_Lai
Email: Charles_Lai@daixiaomi.com
"""
import os
import sys
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import auc, roc_curve, classification_report, confusion_matrix, roc_auc_score
from .dataset_summary import DatasetSummary
import random
import logging
import pandas as pd
import seaborn as sns
import operator
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from utils import encode_to_utf8, exe_time, log_format
from utils.date_time import get_today_string, get_day_before_string, convert_string_to_datetime, \
    get_day_after_string
from src.stat_model.model_utils import ks_test, ks_val
import lightgbm as lgb
import xgboost as xgb

from src.stat_model.feature import R360RiskData
import matplotlib

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../.." % root)
sys.path.append("%s/.." % root)
sys.path.append("%s/../../.." % root)
sys.path.append(u"{0:s}".format(root))
logger = logging.getLogger('lgb')
log_format()

train_message_path = root + "/../output/additional_feature_key_lgb_exp_message.txt"
vali_message_path = root + "/../validation/validation_feature_key_lgb_exp_message.txt"

# LGB_estimators = [x for x in range(125, 175, 25)]
LGB_estimators = [200]
print(LGB_estimators)
LGB_learning_rate = [i for i in np.linspace(0.01, 0.21, 10)]
# LGB_learning_rate = [0.1]
LGB_l2_lambda = [i for i in np.linspace(1e-5, 1e-3, 5)]
# LGB_l2_lambda = [1e-5]
LGB_l1_alpha = [i for i in np.linspace(1e-5, 1e-3, 5)]
# LGB_l1_alpha = [1e-5]
print(LGB_learning_rate)
# LGB_tree_depth = [4, 5]
LGB_tree_depth = [4]

today = get_today_string()


class R360Demo(object):
    def __init__(self):
        self.param_list = self.get_grid_search_params_list()
        self.param_grid = self.get_params_grid()
        pass

    def get_grid_search_params_list(self):
        """

        params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_iterations':10,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'bagging_seed':17,
        'lambda_l1':0.1,
        'lambda_l2':0.1,
        'verbose': 1
    }
        :return: 循环的参数列表
        """
        param_list = list()
        for est in LGB_estimators:
            for dep in LGB_tree_depth:
                for lrate in LGB_learning_rate:
                    for l1_alpha in LGB_l1_alpha:
                        for l2_lambda in LGB_l2_lambda:
                            param = dict()
                            param = {
                                'boosting_type': 'dart',
                                'objective': 'binary',
                                'metric': {'binary_logloss', "auc"},
                                'num_leaves': 255,
                                'feature_fraction': 0.9,
                                'bagging_fraction': 0.8,
                                "min_data_in_leaf": 10,
                                "min_sum_hessian_in_leaf":20,
                                "max_bin": 1024,
                                'bagging_freq': 5,
                                'verbose': -1,
                                'verbosity': -1,
                            "num_threads": 4}
                            param['max_depth'] = dep
                            param['n_estimators'] = est
                            param['l2_lambda'] = l2_lambda
                            param['l1_alpha'] = l1_alpha
                            param['learning_rate'] = lrate
                            param_list.append(param)
        return param_list

    def get_params_grid(self):
        params_grid = {'boosting_type': ['gbdt', 'dart'], 'objective': 'binary', 'metric': 'binary_logloss',
                       'num_leaves': [2 ** n - 1 for n in range(5, 7)], 'feature_fraction': 0.9,
                       'bagging_fraction': 0.8,
                       'bagging_freq': 5, 'verbose': -1}
        params_grid['max_depth'] = LGB_tree_depth
        params_grid['n_estimators'] = LGB_estimators
        params_grid['l2_lambda'] = LGB_l2_lambda
        params_grid['l1_alpha'] = LGB_l1_alpha
        params_grid['learning_rate'] = LGB_learning_rate
        return params_grid

    def __split_feature_label(self, feature_label):
        """
        分开特征和标签
        :param feature_label:
         |uid|features|...|label|
         |---|--------|---|-----|
         |312|23333333|...|  1  |
        :return:
        """
        label_df = feature_label['label']
        del feature_label['label']
        feature = np.array(feature_label)
        return feature, label_df

    @exe_time
    def lgb_class_by_feature(self, feature_label, model_path='', feature_type='r360'):
        """
        对数据进行分类
        :param feature_label: 带feature+label的数据，用于train 和 cv
        :param model_path:
        :param feature_type: 为了区分路径
        :return:
        """
        # 去除特征列表：1. 全为0或全为1， 2. 与其他某几特征重复， 3. 0或1比例小
        # 百分比相关特征，需要log transform
        # 时间过早的数据不宜在进入模型，可调的参数，目前这批数据start设为40000-45000效果最佳
        start = 0
        end = 0
        # 读入数据
        data_summary = DatasetSummary()
        train_df, vali_df = data_summary.get_random_train_validation_set(feature_label)
        x_train, y_train = self.__split_feature_label(train_df)
        x_vali, y_vali = self.__split_feature_label(vali_df)
        key_message_path = train_message_path
        # 数据集标准化
        # scaler = StandardScaler()
        print("数据集标准化 的 训练数据形状", x_train.shape)
        try:
            scaler = joblib.load(model_path + '_scaler.pkl')
            tmp_gbm = joblib.load(model_path + "_little.pkl")
            xtr = scaler.transform(x_train)
        except Exception as e:
            logger.exception(e)
            scaler = StandardScaler()
            xtr = scaler.fit_transform(x_train)
            joblib.dump(scaler, model_path + '_scaler.pkl')
        # new_feature = scaler.transform(np.array(new_feature_df))
        des_string = data_summary.print_label_dataset_detail(y_train, y_vali, y_vali)
        with open(key_message_path, 'a') as fo:
            fo.write(des_string + "\n")
            print(des_string)
        """提前准备报告数据"""
        # check 最后的结果
        xva = scaler.transform(x_vali)
        dtr = lgb.Dataset(xtr, label=y_train, free_raw_data=False)
        dva = lgb.Dataset(xva, label=y_vali, reference=dtr, free_raw_data=False)
        # dte = lgb.Dataset(xte)
        max_ks = 0
        max_n_est = 0
        max_learning_rate = 0
        max_max_depth = 0
        max_ks_test = 0
        max_score = 0
        best_report = 0
        best_clf = 0
        best_param = dict()
        ks_record_path = root+"/../feature/data/ks_lgbt_{0}.txt".format(feature_type)
        # lgb 分类器
        for param in self.param_list:
            params_string = str(param)
            # 给参数列表plst赋值,并根据dtr进行训练,结果保存到clf,再对dte进行测试,结果保存到result
            clf = lgb.train(param, dtr, verbose_eval=False, valid_sets=dva)
            result = clf.predict(xva)
            # result是模型的预测值,y_validation是真实值,求出max_ks & ks_score,返回到ks_test_vali,并写入txt
            ks_vali, threshold_vali = ks_val(result, y_vali, return_threshold=True)
            # predict_label就是对result按照threshold_test切分,大1小0

            # 记录取得最大ks值的参数(max_ks, max_n_est, max_learning_rate, max_max_depth, max_score, best_clf)
            if ks_vali > max_ks:
                max_ks = ks_vali
                max_score = threshold_vali
                best_clf = clf
                best_param = param
                with open(ks_record_path, 'a') as fo:
                    string = 'lgbc test set %f %f %s \n'% (ks_vali, threshold_vali, str(best_param))
                    fo.write(string)
                    logger.info(string)
                predict_label = list()
                for tmp in result:
                    if tmp >= threshold_vali:
                        predict_label.append(1)
                    else:
                        predict_label.append(0)

                result_report = classification_report(y_vali, predict_label,
                                                      target_names=['bad_user', 'good_user'])
                with open(root + '/../output/model_classification_report_{0}.txt'.format(feature_type), 'a') as fo:
                    fo.write(params_string + "\n")
                    fo.write(result_report + "\n")
                # print result_report
                logger.info(result_report)
        little_model_path = model_path + "_little.pkl"
        joblib.dump(best_clf, little_model_path)

        best_ks_params_path = root + '/../output/lgb_best_ks_params_{0}.txt'.format(feature_type)
        with open(best_ks_params_path, 'a') as fo:
            best_ks_string = "MAX KS in Test set: " + str(ks_vali) + 'best ks THRESHOLD: ' + str(
                max_score) + "\t" + "best params : " + str(best_param) + "\n"
            fo.write(best_ks_string)
            fo.write(result_report + "\n")
            print(best_ks_string)
        with open(key_message_path, 'a') as fo:
            fo.write(best_ks_string)
            fo.write(result_report + "\n")
        return best_param

    def r360_predict(self, test_data, model_path):

        x_test, test_fade_label = self.__split_feature_label(test_data)
        scaler = joblib.load(model_path + "_scaler.pkl")
        xte = scaler.transform(x_test)
        best_clf = joblib.load(model_path + "_little.pkl")
        result = best_clf.predict(xte)
        result_df = pd.DataFrame()
        result_df['userid'] = test_data['uid']
        result_df['probability'] = result
        now_time = str(datetime.now())[18]
        result_df.to_csv(root+"/../output/r360_predict_{0}.csv".format(now_time), sep=",", index=False)
