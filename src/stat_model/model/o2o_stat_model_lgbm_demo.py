# -*- coding:utf-8 -*-
__author__ = 'Charles_Lai'
__author_email__ = 'lai.bluejay@gmail.com'

import os
import sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../.." % root)
sys.path.append("%s/.." % root)
sys.path.append("%s/../../.." % root)

sys.path.append("{0:s}".format(root))
import numpy as np
import matplotlib

matplotlib.use('Agg')

# from numpy import *
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import auc, roc_curve, classification_report, confusion_matrix, roc_auc_score
from .dataset_summary import DatasetSummary
import random
import logging
import pandas as pd
import seaborn as sns
import operator
from matplotlib import pyplot as plt

from src.stat_model.feature import FeatureSelector, FeaturePreprocessor, FeatureExtractor
from src.dao import O2OUserDao
from src.stat_model.report import ModelReporter
from utils import encode_to_utf8, exe_time, log_format
from utils.date_time import get_today_string, get_day_before_string, convert_string_to_datetime, get_day_after_string
from src.stat_model.model_utils import ks_test, ks_val

#
logger = logging.getLogger('lgb')
log_format()

train_message_path = root + "/../output/additional_feature_key_lgb_exp_message.txt"
vali_message_path = root + "/../validation/validation_feature_key_lgb_exp_message.txt"

# LGB_estimators = [x for x in range(125, 175, 25)]
LGB_estimators = [200]
print(LGB_estimators)
# LGB_learning_rate = [i for i in np.linspace(0.075, 0.175, 5)]
LGB_learning_rate = [0.1]
# LGB_l2_lambda = [i for i in np.linspace(1e-5, 1e-3, 5)]
LGB_l2_lambda = [1e-5]
# LGB_l1_alpha = [i for i in np.linspace(1e-5, 1e-3, 5)]
LGB_l1_alpha = [1e-5]
print(LGB_learning_rate)
# LGB_tree_depth = [4, 5]
LGB_tree_depth = [4]

today = get_today_string()
# today = "2016-11-17"

"""  KS, PSI的数据集. 模型 统一用训练集+验证集, 作为新的训练样本, 用验证集选择的超参数, 训练新模型M, 作为线上模型.
KS的数据集: test1, 用小数据的模型A.  test_online, 用模型M.
PSI的数据: 用线上模型M, 来预测分数. online数据集, 采用一个月的数据, 分为10天:20天的数据.
切分模型分的数据集: test, 新模型.相当于PSI的test结果数据.

"""

"""
使用sklearn的API进行书写, 可迁移性更高?
"""

class O2OStatModel(object):
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.psi_date = self.__cal_psi_date()
        self.reporter = ModelReporter(begin, end)
        self.o2o_user_dao = O2OUserDao()
        self.param_list = self.get_grid_search_params_list()
        print(self.begin)
        print(self.end)
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
                            params = {
                            'boosting_type': 'gbdt',
                            'objective': 'binary',
                            'metric': 'binary_logloss',
                            'num_leaves': 31,
                            'feature_fraction': 0.9,
                            'bagging_fraction': 0.8,
                            'bagging_freq': 5,
                            'verbose': 0}
                            param['max_depth'] = dep
                            param['n_estimators'] = est
                            param['l2_lambda'] = l2_lambda
                            param['l1_alpha'] = l1_alpha
                            param['learning_rate'] = lrate
                            param_list.append(param)
        return param_list

    def __cal_psi_date(self):
        begin = self.begin
        psi_date = get_day_after_string(10, begin)
        return psi_date

    @exe_time
    def dump_model_file(self, feature_label, best_param, model_path="",
                        model_type="majority", is_export_model_file=True):
        """
        M1数据集上模型训练&选择参数，对于给定参数序列，输出每一组参数对应的训练集KS值
        :param rf_estimators: 随机森林中的决策树数量
        :param rf_max_feature: 每颗决策树在训练时随机采样的特征数量
        :param rf_min_sample_split: 每颗决策树的叶子节点停止分割时最多包含的节点数
        :param feature_file_path: 特征文件路径
        :param result_dir: 结果文件路径
        :param data_set_dir: 数据集路径
        :param data_set_suffix: 数据集后缀
        :param model_dir: 模型文件输出路径
        :param is_export_model_file: 需不需要dump模型文件，调好模型好可设为真
        :param is_export_feature_importance: 需不需要输出特征重要性列表
        :return: 无
        """
        # 去除特征列表：1. 全为0或全为1， 2. 与其他某几特征重复， 3. 0或1比例小
        # 读入数据
        # log变换
        # if len(amount_feature) | len(ratio_feature):
        #     data = get_feature_and_label_one_input_label_file.log_transform(data, reversed_new_feature_pos_to_old_map, \
        #                                                                     amount_feature, ratio_feature)
        label = np.array(feature_label["label"])
        uid_df = pd.DataFrame()
        uid_df["uid"] = feature_label['uid']
        print("训练样本空间大小", feature_label.shape)
        del feature_label["label"]
        del feature_label["uid"]
        new_feature_df = feature_label.iloc[:, :]
        del feature_label

        feature_names = new_feature_df.columns
        new_feature = np.array(new_feature_df)
        del new_feature_df
        data_summary = DatasetSummary()
        x_train, y_train, x_test, y_test, = data_summary.get_train_test_dataset_segments(new_feature, label)
        with open(train_message_path, mode="a") as fo:
            tmp = data_summary.print_one_label_dataset_detail(y_train)
            fo.write("线上模型训练集:" + tmp)
            tmp = data_summary.print_one_label_dataset_detail(y_test)
            fo.write("线上模型测试集:" + tmp)

        del new_feature
        n = len(label)
        apply_id = []
        for i in range(n):
            apply_id.append(i)
        # xrej = data[start + tr_n + va_n + te_n:n, :]
        # yrej = ydata[start + tr_n + va_n + te_n:n]
        # idrej = apply_id[start + tr_n + va_n + te_n:n]
        # 数据集标准化
        scaler = StandardScaler()
        print("标准化数据集的大小", x_train.shape)
        scaler.fit(x_train)
        xtr = scaler.transform(x_train)
        xte = scaler.transform(x_test)
        print("数据集标准化之后的大小", xtr.shape)
        # xrej = scaler.transform(xrej)
        dtr = lgb.Dataset(xtr, label=y_train)
        dte = lgb.Dataset(xte)

        if is_export_model_file:
            joblib.dump(scaler, model_path + '_scaler.pkl')
        # param['num_feature'] = lgb_max_feature[i]
        plst = best_param
        clf = lgb.train(plst, dtr)
        if is_export_model_file:
            # joblib.dump([round((threshold_vali - min_tr) / (max_tr - min_tr) * 620) + 100, min_tr, max_tr],
            #            model_dir + 'randomforestthreshold.0.3.pkl')
            joblib.dump(clf, model_path)
        """大数据集"""
        # check 最后的结果
        uid_list = list(uid_df['uid'])
        o2o_user_dao = O2OUserDao()
        # fpre = FeaturePreprocessor()
        all_phone_label_df = o2o_user_dao.get_user_phone_label_by_uid(uid_list, self.begin, self.end)
        big_new_df = pd.merge(uid_df, all_phone_label_df, on=['uid'], how='left')
        del all_phone_label_df
        # 算一下每一个数据集的长度, 挑选对应长度的数据
        train_len = len(y_train)
        test_len = len(y_test)
        print(test_len)
        train_df = big_new_df[:train_len]
        test_df = big_new_df[train_len:]
        print(test_df.shape)
        del big_new_df
        """提前准备报告数据"""
        # check 最后的结果
        # 预测数据集
        train_result = clf.predict(xtr)
        test_result = clf.predict(xte)
        # 分类报告部分
        # y_validation = label
        self.reporter.output_uid_phone_label_result(train_df, train_result, y_train, model_type, feature_type="train",
                                                    output_type="psi")
        self.reporter.output_uid_phone_label_result(test_df, test_result, y_test, model_type, feature_type="test",
                                                    output_type="psi")

    @exe_time
    def lgbclass_by_feature(self, feature_label, feature_type='overdue', feature_file_path='', model_path='',
                            model_type="majority"):
        """
        M1数据集上模型训练&选择参数，对于给定参数序列，输出每一组参数对应的训练集KS值
        :param rf_estimators: 随机森林中的决策树数量
        :param rf_max_feature: 每颗决策树在训练时随机采样的特征数量
        :param rf_min_sample_split: 每颗决策树的叶子节点停止分割时最多包含的节点数
        :param feature_file_path: 特征文件路径
        :param result_dir: 结果文件路径
        :param model_dir: 模型文件输出路径
        :param is_export_model_file: 需不需要dump模型文件，调好模型好可设为真
        :param is_export_feature_importance: 需不需要输出特征重要性列表
        :return: 无
        """
        # 去除特征列表：1. 全为0或全为1， 2. 与其他某几特征重复， 3. 0或1比例小
        # 百分比相关特征，需要log transform
        # 时间过早的数据不宜在进入模型，可调的参数，目前这批数据start设为40000-45000效果最佳
        start = 0
        end = 0
        # 读入数据

        label = np.array(feature_label["label"])
        uid_df = pd.DataFrame()
        uid_df["uid"] = feature_label['uid']
        print("训练样本空间大小", feature_label.shape)
        del feature_label["label"]
        del feature_label["uid"]
        new_feature_df = feature_label.iloc[:, :]
        del feature_label
        feature_names = new_feature_df.columns
        new_feature = np.array(new_feature_df)

        del new_feature_df

        data_summary = DatasetSummary()
        x_train, y_train, x_validation, y_validation, x_test, y_test, = data_summary.get_dataset_segments(new_feature,
                                                                                                          label)
        del new_feature
        key_message_path = train_message_path
        # 数据集标准化
        # scaler = StandardScaler()
        print("数据集标准化 的 训练数据形状", x_train.shape)
        try:
            scaler = joblib.load(model_path + '_scaler.pkl')
            xtr = scaler.transform(x_train)
        except Exception as e:
            logger.exception(e)
            scaler = StandardScaler()
            xtr = scaler.fit_transform(x_train)
        # new_feature = scaler.transform(np.array(new_feature_df))
        des_string = data_summary.print_label_dataset_detail(y_train, y_validation, y_test)
        with open(key_message_path, 'a') as fo:
            fo.write(des_string + "\n")
            print(des_string)

        # log变换
        # if len(amount_feature) | len(ratio_feature):
        #     data = get_feature_and_label_one_input_label_file.log_transform(data, reversed_new_feature_pos_to_old_map, \
        #                                                                     amount_feature, ratio_feature)
        n = len(label)
        apply_id = []
        for i in range(n):
            apply_id.append(i)
        # xrej = data[start + tr_n + va_n + te_n:n, :]
        # yrej = ydata[start + tr_n + va_n + te_n:n]
        # idrej = apply_id[start + tr_n + va_n + te_n:n]
        """提前准备报告数据"""
        # check 最后的结果
        # xtr = scaler.transform(x_train)
        xte = scaler.transform(x_test)
        xva = scaler.transform(x_validation)
        # xrej = scaler.transform(xrej)
        dtr = lgb.Dataset(xtr, label=y_train)

        dva = lgb.Dataset(xva,label=y_validation, reference=dtr)
        # dte = lgb.Dataset(xte)
        dte = xte

        max_ks = 0
        max_n_est = 0
        max_learning_rate = 0
        max_max_depth = 0
        max_ks_test = 0
        max_score = 0
        best_report = 0
        best_clf = 0
        best_param = dict()
        ks_record_path = "../feature/data/ks_lgbt_{0}.txt".format(feature_type)
        # lgb 分类器
        for param in self.param_list:
            n_estimators = param["n_estimators"]
            dep = param['max_depth']
            lrate = param['learning_rate']
            l2_lambda = param['l2_lambda']
            l1_alpha = param['l1_alpha']
            # param['num_feature'] = lgb_max_feature[i]
            # plst = param.items()
            params_string = str(param)
            print(params_string)
            # 给参数列表plst赋值,并根据dtr进行训练,结果保存到clf,再对dte进行测试,结果保存到result
            clf = lgb.train(param, dtr, verbose_eval=True, valid_sets=dva)
            result = clf.predict(xva)

            # result是模型的预测值,y_validation是真实值,求出max_ks & ks_score,返回到ks_test_vali,并写入txt
            ks_test_vali = ks_val(result, y_validation)
            with open(ks_record_path, 'a') as fo:
                string = 'lgbc test set %d %d %f %f %f %f %f \n' \
                         % (n_estimators, dep, lrate, l1_alpha, l2_lambda,
                            ks_test_vali[0], ks_test_vali[1])
                # print string
                fo.write(string)
                logger.info(string)
            # 这两个参数合一起不就是ks_test_vali?
            ks_test_dynamic, threshold_test = ks_val(result, y_validation)
            with open(ks_record_path, 'a') as fo:
                string = 'lgbc test set %d %d %f %f %f %f %f \n' \
                         % (n_estimators, dep, lrate, l1_alpha, l2_lambda,
                            threshold_test, ks_test_dynamic)
                # print string
                logger.info(string)
                fo.write(string)
            # predict_label就是对result按照threshold_test切分,大1小0
            predict_label = list()
            for tmp in result:
                if tmp >= threshold_test:
                    predict_label.append(1)
                else:
                    predict_label.append(0)

            result_report = classification_report(y_validation, predict_label,
                                                  target_names=['bad_user', 'good_user'])
            with open(root + '/../output/model_classification_report_{0}.txt'.format(feature_type), 'a') as fo:
                fo.write(params_string + "\n")
                fo.write(result_report + "\n")
            # print result_report
            logger.info(result_report)
            # 记录取得最大ks值的参数(max_ks, max_n_est, max_learning_rate, max_max_depth, max_score, best_clf)
            if ks_test_dynamic > max_ks:
                max_ks = ks_test_dynamic
                max_n_est = n_estimators
                max_learning_rate = lrate
                max_max_depth = dep
                max_score = threshold_test
                best_clf = clf
                best_param = param

        little_model_path = model_path + "_little.pkl"
        joblib.dump(best_clf, little_model_path)
        del x_train
        """大数据集"""
        # check 最后的结果
        uid_df['label'] = label
        uid_list = list(uid_df['uid'])
        o2o_user_dao = O2OUserDao()
        # fpre = FeaturePreprocessor()
        all_phone_label_df = o2o_user_dao.get_user_phone_label_by_uid(uid_list, self.begin, self.end)
        big_new_df = pd.merge(uid_df, all_phone_label_df, on=['uid'], how='left')
        del all_phone_label_df
        # 算一下每一个数据集的长度, 挑选对应长度的数据
        # train_len = len(y_train)
        # vali_len = len(y_validation)
        test_len = len(y_test)
        # train_df = big_new_df[:train_len]
        # vali_df = big_new_df[train_len:train_len + vali_len]
        test_df = big_new_df[-test_len:]
        del big_new_df
        """提前准备报告数据"""
        # check 最后的结果
        # 预测数据集
        # train_result = best_clf.predict(dtr)
        # vali_result = best_clf.predict(dva)
        # test_result = best_clf.predict(dte)
        # 分类报告部分
        # y_validation = label
        # 输出PSI的数据
        # self.__output_uid_phone_label_result(train_df, train_result, y_train, model_type, feature_type="train")
        # self.__output_uid_phone_label_result(vali_df, vali_result, y_validation, model_type, feature_type="validation")
        # self.__output_uid_phone_label_result(test_df, test_result, y_test, model_type, feature_type="test")
        # del train_df, vali_df
        # 按照取得最大ks值的参数组合对应的best_clf来预测dva,存到result
        result = best_clf.predict(xte)
        # 分类报告部分
        # 对validation数据集的预测值(result)和真实值, 以及之前循环求出最大ks对应的切分点max_score进行运算,求出ks值存入ks_vali
        ks_vali = ks_test(result, y_test, max_score)
        # auc_test = roc_auc_score(y_test, result)
        # gini_test = 2 * auc_test - 1
        # tmp = '\n在测试集上, 用验证集的阈值进行切分的ks为:{0}, 阈值为:{1}\n测试集上的auc为{2}, Gini系数为{3}\n\n'.format(str(ks_vali), str(max_score),
        #                                                                              str(auc_test), str(gini_test))
        # with open(key_message_path, "a") as fo:
        #     fo.write(tmp + "\n")
        # # print tmp
        # logger.info(tmp)
        # # 对validation数据集的预测值(result)和真实值求max_ks & ks_score
        # new_ks_val, new_ks_threshold = ks_val(result, y_test)
        # tmp = '在测试集上的ks为:{0}, 阈值为{1}'.format(str(new_ks_val), str(new_ks_threshold))
        # with open(key_message_path, "a") as fo:
        #     fo.write(tmp + "\n")
        # # print tmp
        # logger.info(tmp)
        # # 保存最后的结果(用最优参数预测的结果),输出csv,方便后续使用
        # test_df['y_predict'] = list(result)
        # test_df['y_true'] = list(y_test)
        # test_df.to_csv('output/vali_uid_phone_label_result_local_{0}_{1}_{2}.csv'.format(model_type, feature_type,
        #                                                                                  today), index=False)
        self.reporter.model_performance_report(test_df, result, y_test, max_score, model_type, key_message_path,
                                               feature_type)
        self.reporter.record_feature_importance(feature_names, best_clf, feature_type=feature_type,
                                                model_type=model_type)
        # ks—value
        # ks_vali_dynamic = ks_thres[0]
        # # 分割阈值
        # threshold_vali = ks_thres[1]
        ### 按照分割阈值对结果进行分类
        predict_label = list()
        for tmp in result:
            if tmp >= max_score:
                predict_label.append(1)
            else:
                predict_label.append(0)
        result_report = classification_report(y_test, predict_label, target_names=['bad_user', 'good_user'])
        with open(root + '/../output/model_classification_report_{0}.txt'.format(feature_type), 'a') as fo:
            fo.write(result_report + "\n")
        print(result_report)

        best_ks_params_path = root + '/../output/lgb_best_ks_params_{0}.txt'.format(feature_type)
        with open(best_ks_params_path, 'a') as fo:
            best_ks_string = "MAX KS in Test set: " + str(ks_vali) + 'best ks THRESHOLD: ' + str(
                    max_score) + "\t" + "best n_estimators : " + str(
                    max_n_est) + "\t" + "best learning_rate: " + str(
                    max_learning_rate) + "\t" + "best max_depth: " + str(
                    max_max_depth) + "\n"
            fo.write(best_ks_string)
            fo.write(result_report + "\n")
            print(best_ks_string)
        with open(key_message_path, 'a') as fo:
            fo.write(best_ks_string)
            fo.write(result_report + "\n")
        return best_param

    @exe_time
    def due_overdue_model(self, feature_path='data/train_feature_file.txt',
                          model_path='model/model_o2o_stat_lgb_%s_model_20160721_v1.pkl', model_type="majority"):
        feature_extractor = FeatureExtractor()
        # # 数据集初始化
        feature_type = 'overdue'
        # model_path2 = model_path % (feature_type)
        feature_label = feature_extractor.select_feature(feature_path, feature_type, begin=self.begin, end=self.end)
        best_param = self.lgbclass_by_feature(feature_label=feature_label, feature_type=feature_type,
                                             model_path=model_path, model_type=model_type)
        # max_n_est, max_learning_rate, max_max_depth = 150, 0.15, 4
        if self._check_model_cached(model_path):
            # feature_path = 'feature/data/feature_merged_all.txt'
            # 需要再读取一次是因为feature label在训练过程中被删除了
            feature_label = feature_extractor.select_feature(feature_path, feature_type, begin=self.begin, end=self.end)

            # 大训练
            self.dump_model_file(feature_label, best_param=best_param, model_path=model_path, model_type=model_type)
        pass

    @exe_time
    def due_overdue_reject_model(self, feature_path='data/train_feature_file.txt',
                                 model_path='model/model_o2o_stat_lgb_%s_model_20160721_v1.pkl',
                                 model_type="majority"):
        """
        训练时需要第一个KS的数据集.
        :param feature_path:
        :param model_path:
        :param model_type:
        :return:
        """
        feature_type = 'overdue'
        model_path2 = model_path % (feature_type)
        self.due_overdue_model(feature_path, model_path2, model_type)
        # 以下是 当使用逾期数据训练挑选拒绝客户时使用
        best_param = self.lgbclass_add_reject(feature_file_path=feature_path, feature_type='reject',
                                              model_path=model_path2, model_type=model_type)
        # ''' 可以指定feature的路径, feature的label类型是"逾期"还是"拒绝+逾期" '''
        # feature_path = 'feature/data/feature_merged_recent.txt'
        feature_path = root + '/../feature/data/additional_feature_reject_overdue_{0}_{1}_lgb.txt'.format(today, model_type)
        feature_type = 'all'
        model_path = root + '/model_dir/model_o2o_stat_lgb_model_overdue_reject_{0}_{1}.pkl'.format(today, model_type)
        # check_performance_by_model(feature_path, feature_type, model_path, begin='2016-01-01', end='2016-05-31', type='vali')
        # check_performance_by_model(feature_path, feature_type, model_path, begin='2016-01-01', end='2016-05-31', type='test')
        # max_n_est, max_learning_rate, max_max_depth = 150, 0.15, 4
        feature_extractor = FeatureExtractor()
        feature_label = feature_extractor.select_feature(feature_path, feature_type, begin=self.begin, end=self.end)
        if self._check_model_cached(model_path):
            self.dump_model_file(feature_label, best_param=best_param, model_type=model_type, model_path=model_path)
        pass

    def pass_reject_model(self, feature_path='data/train_feature_file.txt',
                          model_path='/model/model_o2o_stat_lgb_%s_model_20160721_v1.pkl'):
        feature_extractor = FeatureExtractor()
        # # 数据集初始化
        # # gen_test_train_validation_set.gen_set('./data/user_chae-at_label_11_18_ll.xlsx', head_name='', id_name='',
        # # v_type=1, suffix='.txt')
        feature_type = 'all'
        model_path = root + model_path % (feature_type)
        # check_performance_by_model(feature_path, feature_type, model_path, begin='2016-01-01', end='2016-05-31')
        # feature_label = feature_extractor.select_feature(feature_path, feature_type)
        feature_label = feature_extractor.select_feature(feature_path, feature_type, begin=self.begin, end=self.end)
        best_param = self.lgbclass_by_feature(feature_label=feature_label, feature_type=feature_type,
                                               model_path=model_path)
        if self._check_model_cached(model_path):
            # feature_path = 'feature/data/feature_merged_all.txt'
            feature_label = feature_extractor.select_feature(feature_path, feature_type, begin=self.begin, end=self.end)
            # 大训练
            self.dump_model_file(feature_label, best_param=best_param, model_path=model_path)

    @exe_time
    def lgbclass_add_reject(self, feature_file_path=root + '/../feature/data/feature_merged_recent.txt', feature_type='overdue',
                             model_path='', is_export_model_file=False,
                            is_export_feature_importance=False, model_type="majority"):
        # 获取拒绝的数据
        logger.info("开始增加拒绝用户的样本")
        feature_extractor = FeatureExtractor()
        feature_pre = FeaturePreprocessor()
        scaler = joblib.load(model_path + '_scaler.pkl')
        reject_feature_path = "../feature/data/additional_feature_reject_{0}_{1}_lgb.txt".format(today, model_type)
        reject_overdue_feature_path = "../feature/data/additional_feature_reject_overdue_{0}_{1}_lgb.txt".format(today,
                                                                                                             model_type)
        key_message_path = train_message_path
        try:
            overdue_reject_data = feature_extractor.read_feature_file(reject_overdue_feature_path)
        except:
            try:
                reject_data = feature_extractor.read_feature_file(reject_feature_path)
            except:
                print('''没有缓存, 再次生成''')
                origin_reject_uid = self.o2o_user_dao.get_user_id_asc(begin=self.begin, end=self.end,
                                                                      user_type='reject')
                origin_len = len(origin_reject_uid)
                # 去掉强规则拒绝的客户, 这样不会影响最后的分布, 这一部分大概去掉了 总用户的8% -10%, 这样总的样本就剩下了全量的90%左右, 未逾期:逾期:拒绝大概为 6:1:2
                strong_rule_reject = self.o2o_user_dao.get_strong_rule_reject_list(self.begin, self.end)
                reject_uid = [uid for uid in origin_reject_uid if uid not in strong_rule_reject]
                new_len = len(reject_uid)
                with open(key_message_path, mode='a') as fo:
                    string = '去掉强规则拒绝的样本数量: %d' % (origin_len - new_len)
                    fo.write(string + "\n")
                    print(string)
                '''改了输入文件之后, 这里的特征选择也要改!!!!!!!!!!!!!!'''
                # 特征选择是啥情况
                feature_pre.select_feature_by_uid_list(reject_uid, output_path=reject_feature_path,
                                                       all_feature_path=feature_file_path)
                reject_data = feature_extractor.read_feature_file(reject_feature_path)
                #  feature, label, feature_names = feature_extractor.select_feature(feature_label_path=reject_feature_path,
                #                                                      feature_type=feature_type)
                # clf = joblib.load(model_path)
                # old_feature_names = clf.booster().feature_names
                # reject_data = reject_data[old_feature_names]
            print("拒绝样本的shape")
            print(reject_data.shape)
            now_reject_uid = list(reject_data['uid'])
            try:
                del reject_data["label"]
            except:
                pass
            try:
                del reject_data['uid']
            except:
                pass
            to_predict_data = reject_data
            print("拒绝样本的数量:", reject_data.shape)
            feature_names = to_predict_data.columns
            print("特征数量为: %d" % len(feature_names))
            # 一个分类器分类
            print('''加载模型,或者在现有的基础上分类''')
            # 归一器
            # scaler = joblib.load(root+'/model/model_o2o_stat_scaler_model_20160623.pkl')
            clf = joblib.load(model_path)
            scaler = joblib.load(model_path + '_scaler.pkl')
            scaler_data = scaler.transform(np.array(to_predict_data))
            del reject_data
            del to_predict_data
            to_pre = lgb.Dataset(scaler_data)
            result = clf.predict(scaler_data)
            # 排序, 获取前50%的阈值. 排序没有返回, 在自身排序
            sorted_result = result[:]
            sorted_result.sort()
            data_length = len(result)
            with open(key_message_path, 'a') as fo:
                string = '拒绝样本数量为:%d' % data_length
                fo.write(string + "\n")
                print(string)
                select_num = int(data_length * 0.3)
                random_num = int(data_length * 0.2)
                threshold = sorted_result[select_num]
                string = "拒绝样本0.3 分位点为{0}, 待增加样本量为两部分, 第一部分是满足条件的uid, {1}个, 第二部分是随机抽取的uid, {2}个。 比例0.3:0.2. " \
                         "一共增加{3}个拒绝样本".format(str(threshold), str(select_num), str(random_num),
                                               str(select_num + random_num))
                fo.write(string + "\n")
                print(string)
                # 读取现有uid
                now_uid = self.o2o_user_dao.get_user_id_asc(begin=self.begin, end=self.end, user_type="pass")
                print(len(now_uid))
                reject_random_uid = list()
                # 存储满足条件的uid 补全label
                # uid有两部分, 第一部分是满足条件的uid, 第二部分是随机抽取的uid。 等量比例0.2:0.2
                for i in range(data_length):
                    tmp = result[i]
                    if tmp < threshold:
                        now_uid.append(now_reject_uid[i])
                    else:
                        '''随机选, 选取等量后结束'''
                        tmp_flag = 0
                        tmp_len = len(reject_random_uid)
                        if tmp_len < random_num:
                            tmp_sup = random.random()
                            # 0.35 保证有足够概率选够0.3的样本
                            if tmp_sup < 0.23:
                                reject_random_uid.append(now_reject_uid[i])
                        else:
                            break
                print(len(now_uid))
                print(len(reject_random_uid))
                # 选出被认为是坏客户的拒绝客户, 并加入训练集
                # 直接把uid丢进去, 内部已经实现了对uid按照时间的排序
                now_uid = now_uid + reject_random_uid
                feature_pre.select_feature_by_uid_list(now_uid, output_path=reject_overdue_feature_path,
                                                       all_feature_path=feature_file_path)
                overdue_reject_data = feature_extractor.read_feature_file(reject_overdue_feature_path)

        overdue_reject_data = overdue_reject_data.fillna(0)

        best_param = self.lgbclass_by_feature(overdue_reject_data, feature_type, feature_file_path=feature_file_path,
                                              model_path=model_path,  model_type=model_type)
        return best_param

    def _check_model_cached(self, model_path):
        try:
            clf = joblib.load(model_path)
        except:
            return True
        return False

    @exe_time
    def check_performance_by_model(self, feature_file_path, feature_type, model_path, type='all',
                                   model_type="minority"):
        with open(vali_message_path, 'a') as fo:
            fo.write("=" * 50 + '\n')
        try:
            current_clf = joblib.load(model_path)
        except Exception as e:
            print(e)
        feature_extractor = FeatureExtractor()
        ''' 可以指定feature的路径, feature的label类型是"逾期"还是"拒绝+逾期" '''
        ''' 选择2016-04-30前的数据做训练  '''
        feature_label = feature_extractor.select_feature(feature_label_path=feature_file_path,
                                                         feature_type=feature_type, begin=self.begin, end=self.end)
        try:
            label = np.array(feature_label["label"])
        except:
            data_length = len(feature_label)
            label = [0 for i in range(data_length)]
        uid_df = pd.DataFrame()
        uid_df["uid"] = feature_label['uid']
        new_feature_df = feature_label.iloc[:, 1:]
        try:
            del new_feature_df['label']
        except:
            pass
        feature_names = new_feature_df.columns
        feature = np.array(new_feature_df)
        del new_feature_df
        # 数据集归一化
        # scaler = StandardScaler()
        scaler = joblib.load(model_path + "_scaler.pkl")
        # scaler.fit(x_train)
        big_vali = scaler.transform(feature)
        dbig_vali = lgb.Dataset(big_vali)
        # check 最后的结果
        uid_df['label'] = label
        uid_list = list(uid_df['uid'])
        new_uid_list = [str(uid) for uid in uid_list]
        print(len(new_uid_list))
        o2o_user_dao = O2OUserDao()
        # fpre = FeaturePreprocessor()
        vali_uid_df = o2o_user_dao.get_user_phone_label_by_uid(uid_list, self.begin, self.end)
        print(len(vali_uid_df))
        new_df = pd.merge(uid_df, vali_uid_df, on=['uid'], how='left')
        print(len(new_df))
        # log变换
        # if len(amount_feature) | len(ratio_feature):
        #     data = get_feature_and_label_one_input_label_file.log_transform(data, reversed_new_feature_pos_to_old_map, \
        #                                                                     amount_feature, ratio_feature)
        # n = len(label)
        # apply_id = []
        # for i in range(n):
        #     apply_id.append(i)

        # if is_export_model_file:
        #     joblib.dump(scaler, model_dir + 'creditlgbscaler.1.2.pkl')
        max_ks = 0
        max_n_est = 0
        max_learning_rate = 0
        max_max_depth = 0
        max_ks_test = 0
        best_clf = current_clf
        best_report = ''
        ks_record_path = "../feature/data/ks_lgbt_{0}.txt".format(feature_type)
        result = best_clf.predict(big_vali)
        # 分类报告部分
        # y_validation = label
        ks_vali, new_ks_threshold = ks_val(result, label)
        new_df["y_true"] = label
        new_df["y_predict"] = result

        # graph_ks_curve(result, label, 'validation/lgbt_ks_curve.jpg')

        print('在验证集上的ks为: %f' % ks_vali)

        self.reporter.model_performance_report(new_df, result, y_test=label, max_score=new_ks_threshold,
                                               model_type=model_type, key_message_path=vali_message_path,
                                               feature_type="all")
        self.reporter.record_feature_importance(feature_names, best_clf, feature_type=feature_type,
                                                model_type=model_type)
        # ks—value
        # ks_vali_dynamic = ks_thres[0]
        # # 分割阈值
        # threshold_vali = ks_thres[1]

        ### 按照分割阈值对结果进行分类
        predict_label = list()
        for tmp in result:
            if tmp >= new_ks_threshold:
                predict_label.append(1)
            else:
                predict_label.append(0)

        result_report = classification_report(label, predict_label, target_names=['bad_user', 'good_user'])
        best_ks_params_path = root + '/../validation/lgb_best_ks_params_{0}.txt'.format(feature_type)
        with open(best_ks_params_path, 'a') as fo:
            best_ks_string = "MAX KS in validation set: " + str(ks_vali) + 'best ks THRESHOLD: ' + str(
                    new_ks_threshold) + "\t" + "best n_estimators : " + str(
                    max_n_est) + "\t" + "best learning_rate: " + str(
                    max_learning_rate) + "\t" + "best max_depth: " + str(
                    max_max_depth) + "\n"
            fo.write(best_ks_string)
            fo.write(result_report + "\n")
            print(best_ks_string)
        with open(vali_message_path, 'a') as fo:
            fo.write(best_ks_string)
            fo.write(result_report + "\n")

    @exe_time
    def get_psi_source_data(self, model_path, feature_file_path, model_type):
        with open(vali_message_path, 'a') as fo:
            fo.write("=" * 50 + '\n')

        feature_extractor = FeatureExtractor()
        ''' 可以指定feature的路径, feature的label类型是"逾期"还是"拒绝+逾期" '''
        feature_type = "all"
        online_feature_label = feature_extractor.select_feature(feature_label_path=feature_file_path,
                                                                feature_type=feature_type, begin=self.begin,
                                                                end=self.psi_date, is_psi=True)
        online_feature_label2 = feature_extractor.select_feature(feature_label_path=feature_file_path,
                                                                 feature_type=feature_type, begin=self.psi_date,
                                                                 end=self.end, is_psi=True)
        self.reporter.output_psi_data(online_feature_label, model_path, feature_type="online1", model_type=model_type)
        self.reporter.output_psi_data(online_feature_label2, model_path, feature_type="online2", model_type=model_type)


if __name__ == "__main__":
    # randomforest模型参数，如需调参可以用数组形式输入
    # RF_estimators = range(150, 400, 50)
    # RF_max_feature = range(4,10)
    # RF_min_sample_split = range(9,14)
    # print "新疆少数民族模型"
    # begin = "2016-03-01"
    # end = "2016-09-30"
    # model = O2OStatModel(begin, end)
    # # with open(train_message_path, 'a') as fo:
    # #     fo.write("=" * 50 + '\n')
    # # # # # 数据集初始化
    # # with open(train_message_path, 'a') as fo:
    # #     sss = "=" * 25 + "少数民族模型" + "=" * 25 + '\n'
    # #     fo.write(sss)
    # #     print sss
    # # # 少数民族模型
    # model_type = "minority"
    # feature_path = "../feature/data/minority_feature_due_overdue_label.txt"
    # feature_type = 'overdue'
    # model_path = 'model_dir/model_o2o_stat_lgb_%s_model_{0}_{1}.pkl'.format(today, model_type)
    # model.due_overdue_reject_model(feature_path, model_path, model_type)
    # model_path = model_path % (feature_type)
    # model.reporter.get_check_split_score_data(feature_path, model_path, model_type)
    #
    # begin = '2016-10-01'
    # end = '2016-10-15'
    # model = O2OStatModel(begin, end)
    # current_model_path = root + '/model_dir/model_o2o_stat_lgb_model_overdue_reject_0808.pkl'
    # minority_model_path = root + "/model_dir/model_o2o_stat_lgb_model_overdue_reject_{0}_minority.pkl".format(today)
    # #
    # '''验证模型在最新数据上的效果'''
    # vali_feature_file = root + '/../feature/data/vali_feature_label.txt'
    # minority_path = root + "/../feature/data/ks_minority_feature_due_overdue_label.txt"
    # feature_type = 'all'
    # with open(vali_message_path, 'a') as fo:
    #     fo.write("=" * 25 + "少数民族模型" + "=" * 25 + '\n')
    # print "验证少数民族模型在最新数据上的效果"
    # model.check_performance_by_model(feature_file_path=minority_path, feature_type=feature_type,
    #                                  model_path=minority_model_path, type='all')

    # '''现有线上模型的效果验证'''
    # # model_path = 'model/lgb_0225.pickle'
    # with open(vali_message_path, 'a') as fo:
    #     fo.write("=" * 25 + "现有模型少数民族" + "=" * 25 + '\n')
    # print "验证现有线上模型在最新数据上的效果"
    # model.check_performance_by_current_model(minority_path, feature_type='reject', model_path=current_model_path)
    # with open(vali_message_path, 'a') as fo:
    #     fo.write("=" * 25 + "现有模型汉族" + "=" * 25 + '\n')
    # model.check_performance_by_current_model(han_nation_path, feature_type='reject', model_path=current_model_path)
    # print "验证现有线上模型在全部民族数据上的效果"
    # with open(vali_message_path, 'a') as fo:
    #     fo.write("=" * 25 + "现有模型全部客户" + "=" * 25 + '\n')
    # model.check_performance_by_current_model(vali_feature_file, feature_type='reject', model_path=current_model_path)

    # begin = '2016-10-01'
    # end = '2016-10-31'
    # model = O2OStatModel(begin, end)
    # current_model_path = root + '/model/model_o2o_stat_lgb_model_overdue_reject_0808.pkl'
    # # han_model_path = root + "/model/model_o2o_stat_lgb_model_overdue_reject_2016-11-17_majority.pkl"
    # # minority_model_path = root + "/model/model_o2o_stat_lgb_model_overdue_reject_2016-11-17_minority.pkl"
    #
    # '''仅用于获取用于计算psi的数据集'''
    # minority_path = root + "/data/vali_minority_feature_due_overdue_label.txt"
    # feature_type = 'all'
    # print "验证少数民族模型的psi的数据"
    # model.get_psi_source_data(feature_file_path=minority_path, model_type="minority", model_path=minority_model_path)

    # """==============================================="""
    print("汉民族模型")
    begin = "2016-09-20"
    end = "2016-09-30"
    model = O2OStatModel(begin, end)
    with open(train_message_path, 'a') as fo:
        fo.write("=" * 50 + '\n')
    # 汉族模型
    with open(train_message_path, 'a') as fo:
        sss = "=" * 25 + "汉族模型" + "=" * 25 + '\n'
        fo.write(sss)
        print(sss)
    model_type = "majority"
    feature_path = "../feature/data/han_nation_feature_due_overdue_label.txt"
    feature_type = 'overdue'
    model_path = 'model_dir/model_o2o_stat_lgb_%s_model_{0}_{1}.pkl'.format(today, model_type)
    model.due_overdue_reject_model(feature_path, model_path, model_type)
    model_path = model_path % (feature_type)
    model.reporter.get_check_split_score_data(feature_path, model_path)

    # begin = '2016-10-01'
    # end = '2016-10-15'
    # model = O2OStatModel(begin, end)
    # current_model_path = root + '/model_dir/model_o2o_stat_lgb_model_overdue_reject_0808.pkl'
    # han_model_path = root + "/model_dir/model_o2o_stat_lgb_model_overdue_reject_{0}_majority.pkl".format(today)
    #
    # '''验证模型在最新数据上的效果'''
    # vali_feature_file = root + '/../feature/data/vali_feature_label.txt'
    # han_nation_path = root + "/../feature/data/ks_han_nation_feature_due_overdue_label.txt"
    # feature_type = 'all'
    # with open(vali_message_path, 'a') as fo:
    #     fo.write("=" * 25 + "汉族模型" + "=" * 25 + '\n')
    # print "验证汉族模型在最新数据上的效果"
    # model.check_performance_by_model(feature_file_path=han_nation_path, feature_type=feature_type,
    #                                  model_path=han_model_path, type="all", model_type="majority")

    # '''现有线上模型的效果验证'''
    # # model_path = 'model/lgb_0225.pickle'
    # with open(vali_message_path, 'a') as fo:
    #     fo.write("=" * 25 + "现有模型少数民族" + "=" * 25 + '\n')
    # print "验证现有线上模型在最新数据上的效果"
    # model.check_performance_by_current_model(minority_path, feature_type='reject', model_path=current_model_path)
    # with open(vali_message_path, 'a') as fo:
    #     fo.write("=" * 25 + "现有模型汉族" + "=" * 25 + '\n')
    # model.check_performance_by_current_model(han_nation_path, feature_type='reject', model_path=current_model_path)
    # print "验证现有线上模型在全部民族数据上的效果"
    # with open(vali_message_path, 'a') as fo:
    #     fo.write("=" * 25 + "现有模型全部客户" + "=" * 25 + '\n')
    # model.check_performance_by_current_model(vali_feature_file, feature_type='reject', model_path=current_model_path)
    #
    # begin = '2016-10-01'
    # end = '2016-10-31'
    # model = O2OStatModel(begin, end)
    # current_model_path = root + '/model_dir/model_o2o_stat_lgb_model_overdue_reject_0808.pkl'
    # # han_model_path = root + "/model_dir/model_o2o_stat_lgb_model_overdue_reject_2016-11-17_majority.pkl"
    # # minority_model_path = root + "/model_dir/model_o2o_stat_lgb_model_overdue_reject_2016-11-17_minority.pkl"
    #
    # '''仅用于获取用于计算ks, psi的数据集'''
    # han_nation_path = root + "/../feature/data/vali_han_nation_feature_due_overdue_label.txt"
    # feature_type = 'all'
    # print "验证汉族模型的psi的数据"
    # model.get_psi_source_data(feature_file_path=han_nation_path, model_type="majority", model_path=han_model_path)
