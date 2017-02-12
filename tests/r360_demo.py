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
from src.stat_model.feature import R360RiskData
from src.stat_model.model.r360_demo import R360Demo
from utils.date_time import get_today_string

root = os.path.dirname(os.path.abspath(__file__))
sys.path.append("%s/../.." % root)
sys.path.append("%s/.." % root)
sys.path.append("%s/../../.." % root)
sys.path.append(u"{0:s}".format(root))

if __name__ == '__main__':
    today = get_today_string()
    r360_obj = R360RiskData()
    train_data, test_data = r360_obj.get_train_test_data()
    r360_demo = R360Demo()
    model_path = root + "/../src/stat_model/model_dir/r360_lgb_demo_{0}.pkl".format(today)
    r360_demo.lgb_class_by_feature(train_data, model_path)
    r360_demo.r360_predict(test_data, model_path)
