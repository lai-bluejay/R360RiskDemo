# -*- coding:utf-8 -*-

from .config import *


class MongoConfig(object):
    def __init__(self, key):
        self.key = key
        self.__get_taobao_mongo_config()

    def __get_taobao_mongo_config(self):
        self.mongodb_address = model_config.get(self.key, 'db_address')
        self.user = model_config.get(self.key, 'user')
        self.passwd = model_config.get(self.key, 'passwd')
        self.db_name = model_config.get(self.key, 'db_name')