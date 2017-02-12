# -*- coding:utf-8 -*-
from .config import *


class MysqlConfig:
    def __init__(self, config_name):
        self.__get_configurations(config_name)

    def __get_configurations(self, config_name):
        self.host = model_config.get(config_name, 'host')
        self.user = model_config.get(config_name, 'user')
        self.passwd = model_config.get(config_name, 'passwd')
        self.db_name = model_config.get(config_name, 'db_name')
