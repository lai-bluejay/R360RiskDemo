# -*- coding: utf-8 -*-
'''
Auth : Xixiang Feng 
Time : 2014-09-04 10:47:30

this program is to calculate the value of KS.
'''
import os
import numpy as np
import operator
import matplotlib as plt
import pandas as pd
import os
import sys
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../.." % root)
sys.path.append("%s/.." % root)
sys.path.append("%s/../../.." % root)
sys.path.append(u"{0:s}".format(root))

'''
@param script: list format, the predicted value(score) of your dataset
@param label_ks: list format, the true label of your dataset
@return the value of KS
'''


def ks_val(data, label_ks, return_threshold=False):
    '''

    :param data: predict result
    :param label_ks: true label
    :param return_threshold: 是否返回ks最大时的阈值
    :return: max ks and  ks threshold
    '''
    data = list(data)
    label_ks = list(label_ks)
    min_score = min(data)
    max_score = max(data)

    res = []
    max_ks = 0
    bad = len(label_ks) - sum(label_ks)
    good = sum(label_ks)
    ks_score = 0
    ks_bad = 0
    ks_good = 0
    for i in np.linspace(min_score, max_score, 50):
        val = [[data[j], label_ks[j]] for j in range(len(data)) if data[j] < i]
        good_now = sum([val[k][1] for k in range(len(val))])
        bad_now = len(val) - good_now
        # bad_now = len([val[k][1] for k in range(len(val)) if val[k][1] == 0])
        # res.append([i, good_now / float(good), bad_now / float(bad)])
        if good == 0:
            tmp_ks_good = 0
        else:
            tmp_ks_good = good_now / float(good)
        if bad == 0:
            tmp_ks_bad = 0
        else:
            tmp_ks_bad = bad_now / float(bad)
        ks_now = abs(tmp_ks_good - tmp_ks_bad)
        if ks_now > max_ks:
            ks_score = i
            ks_good = tmp_ks_good
            ks_bad = tmp_ks_bad
            max_ks = max(max_ks, ks_now)
    # with open(root+'/../output/ks_val.txt', 'a') as fo:
    #     string = "the value of KS: %f, KS_score: %f, good ratio: %f, bad ratio: %f \n" % (
    #     max_ks, ks_score, ks_good, ks_bad)
    #     # print(string)
    #     fo.write(string)
    if return_threshold is True:
        return max_ks, ks_score
    else:
        return max_ks


def ks_test(data, label_ks, ks_score):
    '''
    可以跟上一个合并....
    '''
    bad = len(label_ks) - sum(label_ks)
    good = sum(label_ks)
    val = [[data[j], label_ks[j]] for j in range(len(data)) if data[j] < ks_score]
    good_now = sum([val[k][1] for k in range(len(val))])
    bad_now = len(val) - good_now
    if good == 0:
        tmp_ks_good = 0
    else:
        tmp_ks_good = good_now / float(good)
    if bad == 0:
        tmp_ks_bad = 0
    else:
        tmp_ks_bad = bad_now / float(bad)
    ks_now = abs(tmp_ks_good - tmp_ks_bad)
    # bad_now = len([val[k][1] for k in range(len(val)) if val[k][1] == 0])

    # ks_now = abs(bad_now / float(bad) - good_now / float(good))
    print("the value of KS in the test set: %f" % (ks_now))
    return ks_now


'''
@param output_predict: list format, the predicted value(score) of your dataset
@param true_label: list format, the true label of your dataset
@return the value of AUC
'''


def auc(output_predict, true_label):
    combine = [[output_predict[i], true_label[i]] for i in range(len(true_label))]
    data = sorted(combine, key=operator.itemgetter(0))

    bad = len(true_label) - sum(true_label)
    good = sum(true_label)

    cnt = 0
    for i in range(len(true_label) - 1, -1, -1):
        if data[i][1] == 1:
            cnt += i

    res = (cnt - good * (good - 1) / 2.0) / float(bad * good)
    print("the value of AUC", res)
    return res


'''
@param predStrengths: list format, the predicted value(score) of your dataset
@param classLabels: list format, the true label of your dataset
@return the value of AUC, and the picture of AUC
'''


def plotROC(predStrengths, classLabels):
    predStrengths = np.array(predStrengths)

    cur = (1.0, 1.0)  # cursor
    ySum = 0.0  # variable to calculate AUC
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas);
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = np.argsort(predStrengths, axis=0)  # get sorted index, it's reverse

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point

    for index in sortedIndicies.tolist():
        if classLabels[index[0]] == 1.0:
            delX = 0;
            delY = yStep;
        else:
            delX = xStep;
            delY = 0;
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate');
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)


# 获取模型预测分值分布情况函数
def cal_score_distribution(y_test, y_predict, pos_label=1, neg_label=0, gap_array=None, outFile=None):
    result = list()

    check = pd.DataFrame({'test': y_test, 'predict': y_predict})
    if gap_array is None:
        gap_array = np.arange(0, 1, 0.01)

    pos_total = (check.test == pos_label).value_counts()[True]
    for i in range(len(gap_array) - 1):
        try:
            cnt_pos = ((check.predict >= gap_array[i]) & (check.predict < gap_array[i + 1]) & (
            check.test == pos_label)).value_counts()[True]
        except KeyError:
            cnt_pos = 0

        try:
            cnt_neg = ((check.predict >= gap_array[i]) & (check.predict < gap_array[i + 1]) & (
            check.test == neg_label)).value_counts()[True]
        except KeyError:
            cnt_neg = 0

        try:
            pos_inner_cent = float(cnt_pos) / (cnt_pos + cnt_neg)
        except ZeroDivisionError:
            pos_inner_cent = 0

        try:
            pos_cent = float(cnt_pos) / (pos_total)
        except ZeroDivisionError:
            pos_cent = 0

        result.append(
            ['%.2f - %.2f' % (gap_array[i], gap_array[i + 1]), cnt_pos + cnt_neg, cnt_pos, cnt_neg, pos_inner_cent,
             pos_cent])

    # 将结果保存到文件
    if outFile is not None:
        with open(outFile, 'w') as f:
            for score_arange, cnt_total, cnt_pos, cnt_neg, pos_inner_cent, pos_cent in result:
                f.write(
                    '%s,%d,%d,%d,%.2f,%.2f\n' % (score_arange, cnt_total, cnt_pos, cnt_neg, pos_inner_cent, pos_cent))

    return result


# 获取模型预测分值分布情况函数
'''示例代码'''
# gap_array = np.arange(0,1.0001,0.05)
# distribution_result = cal_score_distribution(y_test,y_scores,pos_label=1,neg_label=0,gap_array=gap_array)
# log_str += 'score_distribution\n'
# log_str += 'score_arange,cnt_total,cnt_pos,cnt_neg,pos_inner_cent,pos_cent\n'
# for score_arange,cnt_total,cnt_pos,cnt_neg,pos_inner_cent,pos_cent in distribution_result:
#     log_str += '%s,%d,%d,%d,%.2f,%.2f\n' % (score_arange,cnt_total,cnt_pos,cnt_neg,pos_inner_cent,pos_cent)
#
# if sys_kind == 'Windows':
#     cur_df = pd.DataFrame(distribution_result)
#     cur_df.columns = ['score_arange','cnt_total','cnt_pos','cnt_neg','pos_inner_cent','pos_cent']
#     cur_df.set_index('score_arange')
#     cur_df = cur_df[['pos_inner_cent','pos_cent']]
#     cur_df.plot(kind='bar',title='score distribution',subplots=True)
