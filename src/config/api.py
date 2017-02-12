# -*- coding:utf-8 -*-

from .config import *


class ApiConfig(object):
    def __init__(self):
        pass

    def __get_phonetags_configurations(self):
        self.phone_api = model_config.get('phone_api', 'url')

    def __get_idcard_configurations(self):
        self.idcard_api = model_config.get('idcard_api', 'url')
