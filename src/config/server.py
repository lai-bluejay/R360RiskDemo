# -*- coding:utf-8 -*-

from .config import *


class ServerConfig:
    def __init__(self):
        self.__get_configurations()

    def __get_configurations(self):
        self.port = model_config.get('server-config', 'port')
        self.version = model_config.get('server-config', "version")
