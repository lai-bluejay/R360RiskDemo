#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
R360_risk.config.py was created on 12/02/2017.
Author: LaiHongchang
Email: lai.bluejay@gmail.com
"""
import os
import configparser

root = os.path.dirname(__file__)


class ModelConfig(object):
    def __init__(self, method):
        """
        输入 test 或者是 real
        :param method:
        :return:
        """
        if method != 'test' and method != "real" and method != "local":
            exit("please init config with test or real")
        config = configparser.ConfigParser()
        config.readfp(open("{0}/{1}/defaults.cfg".format(root, method)))
        self.config = config

    def get(self, key, field):
        return self.config.get(key, field)

model_config = ModelConfig("local")
