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


import matplotlib
matplotlib.use('Agg')
from numpy import *
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from src.stat_model.model.dataset_summary import DatasetSummary
from sklearn.externals import joblib
from .metrics import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import operator
from utils import encode_to_utf8, exe_time

key_message_path = root + "/output/additional_feature_key_xgb_exp_message.txt"

def plot_xgb_roc(feature, label, feature_names, xgb_estimators=200, xgb_learning_rate=0.2, xgb_tree_depth=3, result_dir='',
                 data_set_dir='', data_set_suffix='', model_path='', is_export_model_file=False,
                 is_export_feature_importance=False):
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
    data_summary = DatasetSummary()
    x_train, y_train, x_test, y_test, x_validation, y_validation = data_summary.get_dataset_segments(feature, label)
    data_summary.print_label_dataset_detail(y_train, y_validation, y_test)

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

    # 数据集归一化
    # scaler = StandardScaler()
    # scaler.fit(x_train)
    scaler = joblib.load(model_path+'_scaler.pkl')
    xtr = scaler.transform(x_train)
    xte = scaler.transform(x_test)
    xva = scaler.transform(x_validation)
    # xrej = scaler.transform(xrej)
    dtr = xgb.DMatrix(xtr, label=y_train, feature_names=feature_names)
    dva = xgb.DMatrix(xva, feature_names=feature_names)
    dte = xgb.DMatrix(xte, feature_names=feature_names)



    # XGB 分类器
    print('XGBclass validation set n_estimators max_depth learning_rate max_feature thr ks') \
        # RandomForest
    param = {'silent': 1, 'objective': 'binary:logistic', 'nthread': 4}
    param['bst:max_depth'] = xgb_tree_depth
    param['bst:eta'] = xgb_learning_rate
    # param['num_feature'] = xgb_max_feature[i]
    plst = list(param.items())
    # clf = xgb.train(plst, dtr, xgb_estimators)
    clf = joblib.load(model_path)
    # classifier = clf.fit(xtr, ytr)
    # print len(classifier.oob_prediction_)
    # print classifier.feature_importances_
    # savetxt("np_d313_rf_featureimportance.txt",classifier.feature_importances_)
    # result = classifier.decision_function(xtr)
    # min_tr = min(result)
    # max_tr = max(result)
    # threshold_train = get_ks(result, ytr, -1)[1]
    validation_result = clf.predict(dva)
    output_list = [','.join([str(i), str(j)]) for i, j in zip(validation_result, y_validation)]
    with open('output/xgb_overdue5_validation_result.txt', "w+") as fo:
        output_string = '\n'.join(output_list)
        fo.write(output_string)
    importance = clf.get_fscore(fmap='output/xgb.fmap')
    importance = sorted(list(importance.items()), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    plt.gcf().savefig('output/feature_importance_xgb.png')
    if is_export_feature_importance:
        feature_file_path = 'output/overdue_feature_importance.txt'
        # feature_importance_output = open('RF_feature_importance_output.txt', 'w')
        # importance_array, shape = clf.feature_importances_
        feature_importance = importance
        # for f, v in enumerate(feature_importance):
        #     print "%f\t%s" % (v, feature_name_list[reversed_new_feature_pos_to_old_map[f]])
        with open(feature_file_path, "w+") as fo:
            for im in feature_importance:
                im = encode_to_utf8(im)
                fo.write(str(im)+"\n")

    fig = plt.figure()
    xgb.plot_importance(clf)
    fig.savefig('xgb_reject_feature_importance.jpg')
    ks_thres = ks_val(validation_result, y_validation)
    ks_vali_dynamic = ks_thres[0]
    threshold_vali = ks_thres[1]

    # print 'GBDTc validation set %d %f %d %f' \
    #      % (gbdt_estimators[l], gbdt_learning_rate[k], gbdt_max_feature[i], ks_vali_train)

    test_result = clf.predict(dte)
    # ks_test_train = get_ks(result, yte, threshold_train)[0]
    # print 'GBDTc validation set %d %f %d %f' \
    #       % (gbdt_estimators[l], gbdt_learning_rate[k], gbdt_max_feature[i], ks_test_train)
    ks_test_vali = ks_test(test_result, y_test, threshold_vali)

    ks_test_dynamic, threshold_test = ks_val(test_result, y_test)

    fpr, tpr, thresholds = roc_curve(y_validation, validation_result, pos_label=1)
    fprt, tprt, thresholds_test = roc_curve(y_test, test_result, pos_label=1)
    roc_auc = auc(fpr, tpr)
    roc_auc_test = auc(fprt, tprt)

    fig = plt.figure(num=1,)
    plt.plot(fpr, tpr, lw=1, label='ROC validation curve(area = %0.2f)' % (roc_auc))
    plt.plot(fprt, tprt, lw=1, label = 'ROC test curve(area = %0.2f)' % (roc_auc_test))
    fig.savefig('output/xgbt_roc_curve.jpg')

def graph_ks_curve(y_validation, validation_result, fig_path='output/xgbt_ks_curve.jpg'):
    fpr, tpr, thresholds = roc_curve(y_validation, validation_result, pos_label=1)
    # fprt, tprt, thresholds_test = roc_curve(y_test, test_result, pos_label=1)
    roc_auc = auc(fpr, tpr)
    # roc_auc_test = auc(fprt, tprt)

    fig = plt.figure(num=2,)
    # plt.plot(fpr, tpr, lw=1, label='ROC validation curve(area = %0.2f)' % (roc_auc))
    # plt.plot(fprt, tprt, lw=1, label = 'ROC test curve(area = %0.2f)' % (roc_auc_test))
    test_x_axis = np.arange(len(fpr))/float(len(fpr))
    # fig2 = plt.figure()
    # plt.figure(figsize=[6,6])
    plt.plot(fpr, test_x_axis ,color ='r', label = "overdue")
    plt.plot(tpr, test_x_axis, color = 'g', label = "good")
    plt.title('KS curve')
    plt.legend()
    fig.savefig(fig_path)
    # fig2.savefig('output/xgbt_ks_curve.jpg')

def _get_predict_result():
    result = pd.read_csv('output/xgb_overdue5_validation_result.txt', header = None)

    result.columns = ['y_predict', 'y_true']

    y_validation = result['y_true']
    y_prediction = result['y_predict']
    return y_validation, y_prediction



def _check_model_cached(model_path):
    try:
        clf = joblib.load(model_path)
    except:
        return True
    return False



def create_feature_map(f_path, features):
    outfile = open(f_path, 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

if __name__ == '__main__':
    #
    ''' 可以指定feature的路径, feature的label类型是"逾期"还是"拒绝+逾期" '''
    # feature_path = 'feature/data/feature_merged_recent.txt'
    feature_path = 'data/feature_reject_overdue.txt'
    feature_type = 'reject'

    # feature, label, feature_names = feature_label.select_feature(feature_path, feature_type)
    # create_feature_map(feature_names)
    # feature, label = feature_label.get_feature_label()
    # 逾期+ 拒绝中选择被判断为逾期的数据, 最好参数为 100,0.175, 5
    # 只选择逾期训练, 最好情况为 100, 0.175, 5

    # plot_xgb_roc(feature, label, feature_names, xgb_estimators=100, xgb_learning_rate=0.175,xgb_tree_depth=5, is_export_feature_importance=True)
    # y_validation, y_prediction = _get_predict_result()
    # graph_roc_curve(y_validation, y_prediction)
    _merge_uid_with_result(feature_path, feature_type)