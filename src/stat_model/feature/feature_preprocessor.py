#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Copyright (C) 2017 yongqianbao.com inc. All rights reserved.
R360_risk.feature_preprocessor was created on 11/02/2017.
Author: Charles_Lai
Email: Charles_Lai@daixiaomi.com
"""

import os
import sys

import pandas as pd

import numpy as np

root = os.path.dirname(os.path.abspath(__file__))
print(root)
sys.path.append("%s/../.." % root)
sys.path.append("%s/.." % root)
sys.path.append("%s/../../.." % root)
sys.path.append(u"{0:s}".format(root))


class R360RiskData(object):
    def __init__(self):
        pass

    def load_sample_feature_data(self):
        user_info_train = pd.read_table(root + "/data/train/user_info_train.txt", sep=",", header=None)
        bill_train = pd.read_table(root + "/data/train/bill_detail_train.txt", sep=",", header=None)
        user_info_test = pd.read_table(root + "/data/test/user_info_test.txt", sep=",", header=None)
        bill_test = pd.read_table(root + "/data/test/bill_detail_test.txt", sep=",", header=None)
        test_label = pd.read_table(root + "/data/test/usersID_test.txt", sep=",", header=None)
        train_label = pd.read_table(root + "/data/train/overdue_train.txt", sep=",", header=None)
        bank_detail_train = pd.read_table(root + '/data/train/bank_detail_train.txt', sep=",", header=None)
        bank_detail_test = pd.read_table(root + '/data/test/bank_detail_test.txt', sep=",", header=None)
        browse_history_train = pd.read_csv(root + '/data/train/browse_history_train.txt', header=None)
        browse_history_test = pd.read_csv(root + '/data/test/browse_history_test.txt', header=None)
        loan_time_train = pd.read_csv(root + '/data/train/loan_time_train.txt', header=None)
        loan_time_test = pd.read_csv(root + '/data/test/loan_time_test.txt', header=None)

        # label
        train_label.columns = ['uid', 'label']
        test_label.columns = ["uid"]
        test_label["label"] = 2

        all_label = pd.concat([train_label, test_label])

        # user_info
        user_info = pd.concat([user_info_train, user_info_test])
        user_info.columns = ["uid", 'sex', 'occupation', 'education', 'marry', 'place']
        # user_info.sex=user_info.sex.apply(lambda x:user_info.sex.value_counts().index[0] if x==0 else x)
        # user_info.occupation=user_info.occupation.apply(lambda x:user_info.occupation.value_counts().index[0]
        # if x==0 else x)
        # user_info.education=user_info.education.apply(lambda x:user_info.education.value_counts().index[0]
        # if x==0 else x)
        # user_info.marry=user_info.marry.apply(lambda x:user_info.marry.value_counts().index[0] if x==0 else x)
        # user_info.place=user_info.place.apply(lambda x:user_info.place.value_counts().index[0] if x==0 else x)
        # dummy
        category_col = ['sex', 'occupation', 'education', 'marry', 'place']
        user_info = self.set_dummies(user_info, category_col)

        # bank
        col_names = ['uid', 'tm_encode', 'trade_type', 'trade_amount', 'salary_tag']
        bank_detail_train.columns = col_names
        bank_detail_test.columns = col_names
        bank_detail = pd.concat([bank_detail_train, bank_detail_test])
        # 求收支情况
        bank_detail_n = (bank_detail.loc[:, ['uid', 'trade_type', 'trade_amount',
                                             'tm_encode']]).groupby(['uid', 'trade_type']).mean()
        bank_detail_n = bank_detail_n.unstack()
        bank_detail_n.columns = ['income', 'outcome', 'income_tm', 'outcome_tm']
        bank_detail_n = bank_detail_n.reset_index()

        bank_detail = (bank_detail.loc[:, ['uid', 'trade_type', 'trade_amount',
                                           'tm_encode']]).groupby(['uid', 'trade_type']).last().unstack()
        bank_detail.columns = ['income', 'outcome', 'income_tm', 'outcome_tm']
        bank_detail = bank_detail.reset_index()

        bank = pd.concat([bank_detail_n.ix[:, :3], bank_detail.ix[:, -2:]], axis=1)

        # browse_history
        col_names = ['uid', 'tm_encode_2', 'browse_data', 'browse_tag']
        browse_history_train.columns = col_names
        browse_history_test.columns = col_names
        browse_history = pd.concat([browse_history_train, browse_history_test])
        browse_history_count = browse_history.loc[:, ['uid', 'browse_data']].groupby(['uid']).sum()
        browse_history_count = browse_history_count.reset_index()
        # loan
        loan_time = pd.concat([loan_time_train, loan_time_test])
        loan_time.columns = ['uid', 'loan_time']

        # bill
        col_names = ['uid', 'time', 'bank_id', 'prior_account', 'prior_repay',
                     'credit_limit', 'account_balance', 'minimum_repay', 'consume_count',
                     'account', 'adjust_account', 'circulated_interest', 'available_balance',
                     'cash_limit', 'repay_state']
        bill_train.columns = col_names
        bill_test.columns = col_names
        bill_all = pd.concat([bill_train, bill_test])
        bill_1 = bill_all.groupby("uid").last().reset_index()
        bill_2 = bill_all.groupby("uid").mean().reset_index()
        bill = pd.concat([bill_1.ix[:, :3], bill_2.ix[:, 3:]], axis=1)

        # 合并数据
        feature_data = pd.merge(user_info, loan_time, on="uid", how="outer")
        feature_data = pd.merge(feature_data, bank, on="uid", how="outer")
        feature_data = pd.merge(feature_data, browse_history_count, on="uid", how="outer")
        feature_data = pd.merge(feature_data, bill, on="uid", how="outer")
        # 构造新特征
        feature_data['tm'] = feature_data['loan_time'] - feature_data['time']
        # feature_data['log_id']=pd.Series(np.log(feature_data.uid.values))
        num_features = ['loan_time', 'browse_data', 'time',
                        'bank_id', 'prior_account', 'prior_repay', 'credit_limit',
                        'account_balance', 'minimum_repay', 'consume_count', 'account',
                        'adjust_account', 'circulated_interest', 'available_balance',
                        'cash_limit', 'repay_state', 'tm']
        for feature in num_features:
            feature_data['r_' + feature] = feature_data[feature].rank(method='max')

        for i in feature_data.keys().values:
            feature_data[i] = feature_data[i].fillna(feature_data[i].value_counts().index[0])

        # 缺失值填充
        feature_data = feature_data.fillna(0)
        feature_data = pd.merge(feature_data, train_label, on="uid", how="outer")
        feature_data = feature_data.fillna(2)
        for feature in num_features:
            feature_data['r_' + feature] = feature_data[feature].rank(method='max')

        for i in feature_data.keys().values:
            feature_data[i] = feature_data[i].fillna(feature_data[i].value_counts().index[0])
        return feature_data

    def get_train_test_data(self):
        all_feature_data = self.load_sample_feature_data()
        train_df = all_feature_data[all_feature_data.label != 2]
        test_df = all_feature_data[all_feature_data.label == 2]
        return train_df, test_df

    def set_dummies(self, s_data, col_name):
        for col in col_name:
            dummy = pd.get_dummies(s_data[col], prefix=col)
            #         data.drop(col,
            #                   axis = 1,
            #                   inplace = True)
            s_data = pd.concat([s_data, dummy], axis=1)
        return s_data


if __name__ == '__main__':
    r360 = R360RiskData()
    train_data, test_data = r360.get_train_test_data()
    print(train_data.shape)
    print(train_data.columns)
    print(test_data.shape)
