# -*- coding:utf-8 -*-
from utils import *
"""
Mongo test数据库中 对于 Artcile Collection 的操作类
"""


class MongoHelper:
    def __init__(self, collection):
        self.logger = logging.getLogger(type(self).__name__)
        self._collection = collection

    def find_one(self, condition):
        """
        只取一条数据
        :param condition: 是一个条件字典，例如{"author_phone": phone}
        :param condition:
        :return:
        """
        try:
            ret_value = self._collection.find_one(condition)
            return ret_value
        except Exception as e:
            self.logger.exception(e)
            return None

    def find_many(self, condition):
        """
        获取同一个用户的多条数据
        :param phone:
        :return:
        """
        try:
            ret_value = self._collection.find(condition)
            return ret_value
        except Exception as e:
            self.logger.exception(e)
            return None

    def update_one(self, condition, update_dict):
        try:
            self._collection.update_one(condition, {"$set": update_dict})
        except Exception as e:
            self.logger.exception(e)

    def insert(self, insert_dict):
        try:
            self._collection.insert(insert_dict)
        except Exception as e:
            self.logger.exception(e)

    def remove(self, condition):
        """
        只能是单个的删除
        :param condition: {”_id“:"XXXX"}
        :return:
        """
        if type(condition) != dict:
            return
        if len(condition) != 1:
            return
        if is_blank(condition.get("_id")):
            return
        self._collection.remove(condition)
