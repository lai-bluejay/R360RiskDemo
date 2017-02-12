# -*- coding:utf-8 -*-

"""
model_o2o_stat_model_v2 --  simple_model_train
This file was created at 17/1/9 BY Charles_Lai
"""

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
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.metrics import auc, roc_curve, classification_report, confusion_matrix, roc_auc_score
from .dataset_summary import DatasetSummary
import random
import logging
import pandas as pd

from src.stat_model.feature import FeatureSelector, FeaturePreprocessor, FeatureExtractor
from src.dao import O2OUserDao
from src.stat_model.report import ModelReporter
from utils import encode_to_utf8, exe_time, log_format
from utils.date_time import get_today_string, get_day_before_string, convert_string_to_datetime, get_day_after_string
from src.stat_model.model_utils import ks_test, ks_val
from .o2o_stat_model_xgb import O2OStatModel

#
logger = logging.getLogger('xgb')
log_format()

train_message_path = root + "/../output/additional_feature_key_xgb_exp_message.txt"
vali_message_path = root + "/../validation/validation_feature_key_xgb_exp_message.txt"

# XGB_estimators = [x for x in range(125, 175, 25)]
XGB_estimators = [200]
print(XGB_estimators)
XGB_max_feature = [543]
# XGB_learning_rate = [i for i in np.linspace(0.075, 0.175, 5)]
XGB_learning_rate = [0.1]
# XGB_l2_lambda = [i for i in np.linspace(1e-5, 1e-3, 5)]
XGB_l2_lambda = [1e-5]
# XGB_l1_alpha = [i for i in np.linspace(1e-5, 1e-3, 5)]
XGB_l1_alpha = [1e-5]
print(XGB_learning_rate)
# XGB_tree_depth = [4, 5]
XGB_tree_depth = [4]

today = get_today_string()
# today = "2016-11-17"

"""  KS, PSI的数据集. 模型 统一用训练集+验证集, 作为新的训练样本, 用验证集选择的超参数, 训练新模型M, 作为线上模型.
KS的数据集: test1, 用小数据的模型A.  test_online, 用模型M.
PSI的数据: 用线上模型M, 来预测分数. online数据集, 采用一个月的数据, 分为10天:20天的数据.
切分模型分的数据集: test, 新模型.相当于PSI的test结果数据.

"""


class SimpleStatModel(O2OStatModel):
    def __init__(self, begin, end):
        super(SimpleStatModel, self).__init__(begin, end)

    @exe_time
    def due_overdue_model(self, feature_path='data/train_feature_file.txt',
                          model_path='model/model_o2o_stat_xgb_%s_model_20160721_v1.pkl', model_type="majority"):
        feature_extractor = FeatureExtractor()
        # # 数据集初始化
        feature_type = 'overdue'
        # model_path2 = model_path % (feature_type)
        # feature_label = feature_extractor.select_feature(feature_path, feature_type, begin=self.begin, end=self.end)
        feature_label = feature_extractor.simple_read_feature_file(feature_path, feature_type, begin=self.begin,
                                                                   end=self.end)
        feature_label = feature_label.fillna(0)
        best_param = self.xgbclass_by_feature(feature_label=feature_label, feature_type=feature_type,
                                              model_path=model_path, model_type=model_type)
        # max_n_est, max_learning_rate, max_max_depth = 150, 0.15, 4
        if self._check_model_cached(model_path):
            # feature_path = 'feature/data/feature_merged_all.txt'
            # 需要再读取一次是因为feature label在训练过程中被删除了
            # feature_label = feature_extractor.select_feature(feature_path, feature_type, begin=self.begin, end=self.end)
            feature_label = feature_extractor.simple_read_feature_file(feature_path, feature_type, begin=self.begin,
                                                                       end=self.end)
            # 大训练
            self.dump_model_file(feature_label, best_param=best_param, model_path=model_path, model_type=model_type)
        pass

    @exe_time
    def due_overdue_reject_model(self, feature_path='data/train_feature_file.txt',
                                 model_path='model/model_o2o_stat_xgb_%s_model_20160721_v1.pkl',
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
        best_param = self.xgbclass_add_reject(feature_file_path=feature_path, feature_type='reject',
                                              model_path=model_path2, model_type=model_type)
        # ''' 可以指定feature的路径, feature的label类型是"逾期"还是"拒绝+逾期" '''
        # feature_path = 'feature/data/feature_merged_recent.txt'
        feature_path = root + '/../feature/data/additional_feature_reject_overdue_{0}_{1}.txt'.format(today, model_type)
        feature_type = 'all'
        model_path = root + '/model_dir/model_o2o_stat_xgb_model_overdue_reject_{0}_{1}.pkl'.format(today, model_type)
        # check_performance_by_model(feature_path, feature_type, model_path, begin='2016-01-01', end='2016-05-31', type='vali')
        # check_performance_by_model(feature_path, feature_type, model_path, begin='2016-01-01', end='2016-05-31', type='test')
        # max_n_est, max_learning_rate, max_max_depth = 150, 0.15, 4
        feature_extractor = FeatureExtractor()
        feature_label = feature_extractor.simple_read_feature_file(feature_path, feature_type, begin=self.begin,
                                                                   end=self.end)
        if self._check_model_cached(model_path):
            self.dump_model_file(feature_label, best_param=best_param, model_path=model_path)
        pass

    def pass_reject_model(self, feature_path='data/train_feature_file.txt',
                          model_path='/model/model_o2o_stat_xgb_%s_model_20160721_v1.pkl'):
        feature_extractor = FeatureExtractor()
        # # 数据集初始化
        # # gen_test_train_validation_set.gen_set('./data/user_chae-at_label_11_18_ll.xlsx', head_name='', id_name='',
        # # v_type=1, suffix='.txt')
        feature_type = 'all'
        model_path = root + model_path % (feature_type)
        # check_performance_by_model(feature_path, feature_type, model_path, begin='2016-01-01', end='2016-05-31')
        # feature_label = feature_extractor.select_feature(feature_path, feature_type)
        feature_label = feature_extractor.select_feature(feature_path, feature_type, begin=self.begin, end=self.end)
        best_param = self.xgbclass_by_feature(feature_label=feature_label, feature_type=feature_type,
                                              data_set_dir='./data/',

                                              model_path=model_path)
        if self._check_model_cached(model_path):
            # feature_path = 'feature/data/feature_merged_all.txt'
            feature_label = feature_extractor.select_feature(feature_path, feature_type, begin=self.begin, end=self.end)
            # 大训练
            self.dump_model_file(feature_label, best_param=best_param, model_path=model_path)

    @exe_time
    def xgbclass_add_reject(self, feature_file_path=root + '/../feature/data/feature_merged_recent.txt', feature_type='overdue',
                            model_path='', is_export_model_file=False, is_export_feature_importance=False,
                            model_type="majority"):
        # 获取拒绝的数据
        logger.info("开始增加拒绝用户的样本")
        feature_extractor = FeatureExtractor()
        feature_pre = FeaturePreprocessor()
        scaler = joblib.load(model_path + '_scaler.pkl')
        reject_feature_path = "../feature/data/additional_feature_reject_{0}_{1}.txt".format(today, model_type)
        reject_overdue_feature_path = "../feature/data/additional_feature_reject_overdue_{0}_{1}.txt".format(today,
                                                                                                             model_type)
        key_message_path = train_message_path
        try:
            # overdue_reject_data = feature_extractor.read_feature_file(reject_overdue_feature_path)
            overdue_reject_data = feature_extractor.simple_read_feature_file(reject_overdue_feature_path,
                                                                             feature_type="all",
                                                                             begin=self.begin, end=self.end)
        except:
            try:
                # reject_data = feature_extractor.read_feature_file(reject_feature_path)
                reject_data = feature_extractor.simple_read_feature_file(reject_feature_path, feature_type,
                                                                         begin=self.begin, end=self.end)
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
                reject_data = feature_extractor.simple_read_feature_file(reject_feature_path, feature_type,
                                                                         begin=self.begin, end=self.end)
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
            to_pre = xgb.DMatrix(scaler_data, feature_names=feature_names)
            result = clf.predict(to_pre)
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
                # overdue_reject_data = feature_extractor.read_feature_file(reject_overdue_feature_path)
                overdue_reject_data = feature_extractor.simple_read_feature_file(reject_overdue_feature_path,
                                                                                 feature_type="all", begin=self.begin,
                                                                                 end=self.end)
        overdue_reject_data = overdue_reject_data.fillna(0)

        best_param = self.xgbclass_by_feature(overdue_reject_data, feature_type, feature_file_path=feature_file_path,
                                              model_path=model_path, model_type=model_type)
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
        dbig_vali = xgb.DMatrix(big_vali, feature_names=feature_names)
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
        #     joblib.dump(scaler, model_dir + 'creditxgbscaler.1.2.pkl')
        max_ks = 0
        max_n_est = 0
        max_learning_rate = 0
        max_max_depth = 0
        max_ks_test = 0
        best_clf = current_clf
        best_report = ''
        ks_record_path = "../feature/data/ks_xgbt_{0}.txt".format(feature_type)
        result = best_clf.predict(dbig_vali)
        # 分类报告部分
        # y_validation = label
        ks_vali, new_ks_threshold = ks_val(result, label)
        new_df["y_true"] = label
        new_df["y_predict"] = result

        # graph_ks_curve(result, label, 'validation/xgbt_ks_curve.jpg')

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

        result_report = classification_report(label, predict_label,
                                              target_names=['bad_user', 'good_user'])
        best_ks_params_path = root + '/../validation/xgb_best_ks_params_{0}.txt'.format(feature_type)
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
    # randomforest模型参数，如需调参可以用数组形式输入+
    """==============================================="""
    print("同盾数据模型")
    begin = "2016-09-24"
    end = "2016-10-01"
    model = SimpleStatModel(begin, end)
    with open(train_message_path, 'a') as fo:
        fo.write("=" * 50 + '\n')
    # 汉族模型
    with open(train_message_path, 'a') as fo:
        sss = "=" * 25 + "同盾数据" + "=" * 25 + '\n'
        fo.write(sss)
        print(sss)
    model_type = "majority"
    feature_path = "../feature/data/tongdun_feature_label.csv"
    feature_type = 'overdue'
    model_path = 'model_dir/tongdun_o2o_stat_xgb_%s_model_{0}_{1}.pkl'.format(today, model_type)
    model.due_overdue_reject_model(feature_path, model_path, model_type)
    # model_path = model_path % (feature_type)
    # model.reporter.get_check_split_`score_data(feature_path, model_path)
