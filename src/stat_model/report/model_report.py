# -*- coding:utf-8 -*-

"""
model_o2o_stat_model_v2 --  model_report
This file was created at 16/12/22 BY Charles_Lai
"""

__author__ = 'Charles_Lai'
__author_email__ = 'lai.bluejay@gmail.com'

import logging
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib
matplotlib.use('Agg')
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import operator

from sklearn.externals import joblib
from sklearn.metrics import auc, roc_curve, classification_report, confusion_matrix, roc_auc_score
import random
import logging
import pandas as pd
import seaborn as sns
import operator
from matplotlib import pyplot as plt

from src.dao import O2OUserDao

from utils import encode_to_utf8, exe_time


from utils.log_utils import log_format, exe_time
from utils.date_time import get_today_string, get_day_before_string
from src.stat_model.model_utils import ks_val, ks_test, create_feature_map

today = get_today_string()
# logger = logging.getLogger("ModelReporter")
class ModelReporter(object):
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.logger = logging.getLogger(__name__)
        pass

    @exe_time
    def model_performance_report(self, new_df, result, y_test, max_score, model_type, key_message_path, feature_type,
                                 output_type="ks"):
        """

        :param new_df:
        :param result:
        :param y_test:
        :param max_score:
        :param model_type:
        :param key_message_path:
        :param feature_type: 特征类型:  overdue , reject.  如果仅仅是逾期模型,就是overdue,  如果有拒绝样本,就是"reject"
        :param output_type: 输出数据类型, ks  或者 psi, 或者 split
        :return:
        """
        new_df['y_true'] = y_test
        new_df["y_predict"] = result
        # 分类报告部分
        # 对validation数据集的预测值(result)和真实值求max_ks & ks_score
        new_ks_val, new_ks_threshold = ks_val(result, y_test)
        auc_test = roc_auc_score(y_test, result)
        gini_test = 2 * auc_test - 1
        tmp = '\n在测试集上的ks为:{0}, 阈值为{1}\n测试集上的auc为{2}, Gini系数为{3}\n\n'.format(str(new_ks_val), str(new_ks_threshold),
                                                                                     str(auc_test), str(gini_test))
        with open(key_message_path, "a") as fo:
            fo.write(tmp + "\n")
        # print tmp
        self.logger.info(tmp)
        self.output_uid_phone_label_result(new_df, result, y_test, model_type, feature_type, output_type)
        pass_reject_string = ""
        due_reject_string = ""
        ### 准备分类讨论的数据
        pass_reject_true = list(new_df.pass_status)
        due_reject_df = new_df[new_df.pass_status == new_df["label"]]
        due_reject_predict = list(due_reject_df['y_predict'])
        due_reject_true = list(due_reject_df['label'])
        due_overdue_df = new_df[new_df.pass_status == 1]
        print(due_overdue_df.shape)
        due_overdue_predict = list(due_overdue_df['y_predict'])
        due_overdue_true = list(due_overdue_df['label'])
        #### 这里包含了通过未逾期, 通过逾期, 拒绝客户三类 . 可用于计算ks
        ###计算通过拒绝的ks吧!!!!!!!
        if feature_type == 'reject':
            pass_reject_ks = ks_test(result, pass_reject_true, max_score)
            # pass_reject_ks, max_score = ks_val(result, pass_reject_true)
            pass_reject_string = " The KS value in pass&reject set is {0}, threshold is {1}".format(str(pass_reject_ks),
                                                                                                    str(max_score))
            # print pass_reject_string
            self.logger.info(pass_reject_string)

            print(due_reject_df.shape)
            due_reject_ks = ks_test(due_reject_predict, due_reject_true, max_score)
            due_reject_string = " The KS value in due&reject set is {0}, threshold is {1}".format(str(due_reject_ks),
                                                                                                  str(max_score))
            # print due_reject_string
            self.logger.info(due_reject_string)
        else:
            pass
        print(due_overdue_df.shape)
        due_overdue_ks = ks_test(due_overdue_predict, due_overdue_true, max_score)
        due_overdue_string = " The KS value in due&overdue set is {0}, threshold is {1}".format(str(due_overdue_ks),
                                                                                                str(
                                                                                                        max_score))

        # print due_overdue_string
        self.logger.info(due_overdue_string)
        with open(key_message_path, 'a') as fo:
            fo.write(pass_reject_string + "\n")
            fo.write(due_reject_string + "\n")
            fo.write(due_overdue_string + "\n")

        # 计算新的ks阈值上的各类ks
        tmp = "计算新的ks阈值上的各类ks"
        with open(key_message_path, "a") as fo:
            fo.write(tmp + "\n")
        print(tmp)
        #### 这里包含了通过未逾期, 通过逾期, 拒绝客户三类 . 可用于计算ks
        ###计算通过拒绝的ks吧!!!!!!!
        if feature_type == 'reject':
            pass_reject_ks = ks_test(result, pass_reject_true, new_ks_threshold)
            pass_reject_string = " The KS value in pass&reject set is {0}, threshold is {1}".format(str(pass_reject_ks),
                                                                                                    str(new_ks_threshold))
            # print pass_reject_string
            self.logger.info(pass_reject_string)

            print(due_reject_df.shape)
            due_reject_ks = ks_test(due_reject_predict, due_reject_true, new_ks_threshold)
            due_reject_string = " The KS value in due&reject set is {0}, threshold is {1}".format(str(due_reject_ks),
                                                                                                  str(new_ks_threshold))
            # print due_reject_string
            self.logger.info(due_reject_string)
        else:
            pass
        print(due_overdue_df.shape)
        due_overdue_ks = ks_test(due_overdue_predict, due_overdue_true, new_ks_threshold)
        due_overdue_string = " The KS value in due&overdue set is {0}, threshold is {1}".format(str(due_overdue_ks),
                                                                                                str(new_ks_threshold))
        # print due_overdue_string
        self.logger.info(due_overdue_string)
        with open(key_message_path, 'a') as fo:
            fo.write(pass_reject_string + "\n")
            fo.write(due_reject_string + "\n")
            fo.write(due_overdue_string + "\n")

        ###各自计算ks和阈值, 通过拒绝的ks吧!!!!!!!
        tmp = "各类数据集计算ks和阈值"
        with open(key_message_path, "a") as fo:
            fo.write(tmp + "\n")
        print(tmp)
        if feature_type == 'reject':
            pass_reject_true = list(new_df.pass_status)
            pass_reject_ks, pass_reject_ks_threshold = ks_val(result, pass_reject_true)
            pass_reject_string = " The KS value in pass&reject set is {0}, threshold is {1}".format(str(pass_reject_ks),
                                                                                                    str(pass_reject_ks_threshold))
            # print pass_reject_string
            self.logger.info(pass_reject_string)
            print(due_reject_df.shape)
            due_reject_ks, due_reject_ks_threshold = ks_val(due_reject_predict, due_reject_true)
            due_reject_string = " The KS value in due&reject set is {0}, threshold is {1}".format(str(due_reject_ks),
                                                                                                  str(due_reject_ks_threshold))
            # print due_reject_string
            self.logger.info(due_reject_string)
        else:
            pass
        print(due_overdue_df.shape)
        due_overdue_ks, due_overdue_ks_threshold = ks_val(due_overdue_predict, due_overdue_true)
        due_overdue_string = " The KS value in due&overdue set is {0}, threshold is {1}".format(str(due_overdue_ks),
                                                                                                str(due_overdue_ks_threshold))
        # print due_overdue_string
        self.logger.info(due_overdue_string)
        with open(key_message_path, 'a') as fo:
            fo.write(pass_reject_string + "\n")
            fo.write(due_reject_string + "\n")
            fo.write(due_overdue_string + "\n")

        # 计算在逾期未逾期上最高的ks的切分点对应的各个的ks值
        max_score = due_overdue_ks_threshold
        tmp = "计算在逾期未逾期上最高的ks的切分点对应的各个的ks值"
        with open(key_message_path, "a") as fo:
            fo.write(tmp + "\n")
        print(tmp)
        if feature_type == 'reject':
            pass_reject_ks = ks_test(result, pass_reject_true, max_score)
            pass_reject_string = " The KS value in pass&reject set is {0}, threshold is {1}".format(str(pass_reject_ks),
                                                                                                    str(max_score))
            # print pass_reject_string
            self.logger.info(pass_reject_string)
            print(due_reject_df.shape)
            due_reject_ks = ks_test(due_reject_predict, due_reject_true, max_score)
            due_reject_string = " The KS value in due&reject set is {0}, threshold is {1}".format(str(due_reject_ks),
                                                                                                  str(max_score))
            # print due_reject_string
            self.logger.info(due_reject_string)
        else:
            pass
        print(due_overdue_df.shape)
        due_overdue_ks = ks_test(due_overdue_predict, due_overdue_true, max_score)
        due_overdue_string = " The KS value in due&overdue set is {0}, threshold is {1}".format(str(due_overdue_ks),
                                                                                                str(max_score))
        # print due_overdue_string
        self.logger.info(due_overdue_string)
        with open(key_message_path, 'a') as fo:
            fo.write(pass_reject_string + "\n")
            fo.write(due_reject_string + "\n")
            fo.write(due_overdue_string + "\n")
        ### 按照分割阈值对结果进行分类
        predict_label = list()
        for tmp in result:
            if tmp >= max_score:
                predict_label.append(1)
            else:
                predict_label.append(0)
        result_report = classification_report(y_test, predict_label,
                                              target_names=['bad_user', 'good_user'])
        print(result_report)
        with open(key_message_path, 'a') as fo:
            fo.write(result_report + "\n")

    def output_uid_phone_label_result(self, new_df, result, y_test, model_type, feature_type="reject",
                                        output_type="psi"):
        """
        保存最后的结果(用最优参数预测的结果),输出csv,方便后续使用
        :param new_df: 待输出的dataframe
        :param result: 预测的结果
        :param y_test:
        :param model_type:
        :param feature_type:
        :return:
        """
        # 保存最后的结果(用最优参数预测的结果),输出csv,方便后续使用
        new_df['y_predict'] = list(result)
        new_df['y_true'] = list(y_test)
        new_df.to_csv(root + '/../output/{0}_{1}_{2}_{3}.csv'.format(model_type, feature_type, output_type, today),index=False)

    def record_feature_importance(self, feature_names, best_clf, feature_type="overdue", model_type="majority"):

        if isinstance(best_clf, lgb.Booster) is False:
            feature_file_path = root + '/../output/{0}_{1}_feature_importance_xgb.txt'.format(model_type, feature_type)
            fpath = root + '/../output/xgb_{0}_{1}.fmap'.format(model_type, feature_type)
            create_feature_map(fpath, feature_names)
            importance = best_clf.get_fscore(fmap=fpath)
            importance = sorted(list(importance.items()), lambda x, y: cmp(x[1], y[1]), reverse=True)
            df = pd.DataFrame(importance, columns=['feature', 'fscore'])
            df['fscore'] = df['fscore'] / df['fscore'].sum()
            plot_df = df.sort_values(by=['fscore'])
            plot_df = plot_df.head(20)

            plt.figure()
            df.plot()
            df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
            plt.title('XGBoost Feature Importance')
            plt.xlabel('relative importance')
            plt.gcf().savefig(root + '/../output/{0}_{1}_feature_importance_xgb.png'.format(model_type, feature_type))
            # feature_importance_output = open('RF_feature_importance_output.txt', 'w')
            # importance_array, shape = clf.feature_importances_
            feature_importance = importance
            # for f, v in enumerate(feature_importance):
            #     print "%f\t%s" % (v, feature_name_list[reversed_new_feature_pos_to_old_map[f]])

            # print feature_importance
            # print type(feature_importance)
            with open(feature_file_path, "w+") as fo:
                for im in feature_importance:
                    # print im
                    # print type(im)
                    im = encode_to_utf8(im)
                    im = [str(iii) for iii in im]
                    if isinstance(im, tuple) or isinstance(im, list):
                        im_str = "\t".join(im)
                    else:
                        im_str = str(im)
                    fo.write(im_str + "\n")
            fig = plt.figure()
            xgb.plot_importance(best_clf)
            fig.savefig(root + '/../output/{0}_{1}_xgb_reject_feature_importance.jpg'.format(model_type, feature_type))
        else:
            feature_file_path = root + '/../output/{0}_{1}_feature_importance_lgb.txt'.format(model_type, feature_type)
            self.logger.info('Calculate feature importances...')
            # feature importances
            print(('Feature importances:', list(best_clf.feature_importance())))
            print(('Feature GAIN importances:', list(best_clf.feature_importance("gain"))))
            with open(feature_file_path, "w+") as fo:

                fo.write('Feature importances:'+"\n")
                fo.write("\n".join([str(i) for i in list(best_clf.feature_importance())]))
                fo.write('Feature GAIN importances:'+"\n")
                fo.write("\n".join([str(i) for i in list(best_clf.feature_importance("gain"))]))
            fig = plt.figure()
            lgb.plot_importance(best_clf)
            fig.savefig(root + '/../output/{0}_{1}_lgb_reject_feature_importance.jpg'.format(model_type, feature_type))

    @exe_time
    def get_check_split_score_data(self, feature_file_path, model_path, model_type="majority"):
        scaler = joblib.load(model_path + '_scaler.pkl')
        best_clf = joblib.load(model_path + "_little.pkl")
        # key_message_path = train_message_path
        tmp_feature_extractor = FeatureExtractor()
        two_month_ago = get_day_before_string(60, date=self.end)
        print(two_month_ago)
        check_split_df = tmp_feature_extractor.load_features(feature_file_path, feature_type="all",
                                                             begin=two_month_ago, end=self.end)
        check_label = np.array(check_split_df["label"])
        tmp_uid_df = pd.DataFrame()
        tmp_uid_df["uid"] = check_split_df['uid']
        print("训练样本空间大小", check_split_df.shape)
        del check_split_df["label"]
        tmp_check_feature_df = check_split_df.iloc[:, 1:]
        del check_split_df
        tmp_feature_names = tmp_check_feature_df.columns
        tmp_feature = np.array(tmp_check_feature_df)

        del tmp_check_feature_df

        tmp_xtr = scaler.transform(tmp_feature)

        # xrej = scaler.transform(xrej)
        tmp_dtr = xgb.DMatrix(tmp_xtr, feature_names=tmp_feature_names)
        """大数据集"""
        # check 最后的结果
        tmp_uid_df['label'] = check_label
        tmp_uid_list = list(tmp_uid_df['uid'])
        o2o_user_dao = O2OUserDao()
        # fpre = FeaturePreprocessor()
        tmp_all_phone_label_df = o2o_user_dao.get_user_phone_label_by_uid(tmp_uid_list, self.begin, self.end)
        big_check_df = pd.merge(tmp_uid_df, tmp_all_phone_label_df, on=['uid'], how='left')
        del tmp_all_phone_label_df
        # 算一下每一个数据集的长度, 挑选对应长度的数据
        """提前准备报告数据"""
        # check 最后的结果
        # 预测数据集
        if isinstance(best_clf, lgb.Booster):
            tmp_train_result = best_clf.predict(tmp_xtr)
        else:
            tmp_train_result = best_clf.predict(tmp_dtr)

        # 分类报告部分
        # tmp_y_validation = label
        self.output_uid_phone_label_result(big_check_df, tmp_train_result, check_label, model_type,
                                             feature_type="two_month", output_type="split")

    def output_psi_data(self, feature_label, model_path, feature_type, model_type):
        """
        输出psi 的数据集
        :param feature_label:
        :param model_path:
        :param feature_type:
        :return:
        """
        try:
            best_clf = joblib.load(model_path)
        except Exception as e:
            print(e)
        label = np.array(feature_label["label"])
        uid_df = pd.DataFrame()
        uid_df["uid"] = feature_label['uid']
        new_feature_df = feature_label.iloc[:, 1:]
        del new_feature_df['label']
        feature_names = new_feature_df.columns
        feature = np.array(new_feature_df)
        del new_feature_df
        # 数据集归一化
        # scaler = StandardScaler()
        scaler = joblib.load(model_path + "_scaler.pkl")
        xtr = scaler.transform(feature)
        dtr = xgb.DMatrix(xtr, feature_names=feature_names)
        """大数据集"""
        # check 最后的结果
        uid_df['label'] = label
        print(uid_df.shape)
        uid_list = list(uid_df['uid'])
        o2o_user_dao = O2OUserDao()
        # fpre = FeaturePreprocessor()
        all_phone_label_df = o2o_user_dao.get_user_phone_label_by_uid(uid_list, self.begin, self.end)
        print(all_phone_label_df.shape)
        big_new_df = pd.merge(uid_df, all_phone_label_df, on=['uid'], how='left')
        print(big_new_df.shape)
        # 预测数据集
        train_result = best_clf.predict(dtr)
        # 分类报告部分
        # y_validation = label
        self.output_uid_phone_label_result(big_new_df, train_result, label, model_type,
                                                    feature_type=feature_type, output_type="psi")