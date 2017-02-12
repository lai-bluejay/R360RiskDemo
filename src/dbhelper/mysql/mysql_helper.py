# -*- coding:utf-8 -*-
from utils import *


class MySQLHelper:
    def __init__(self, db):
        self._db = db
        assert self._db is not None

    def excute(self, sql, *kwargs):
        """
        执行 update, insert, delete 操作
        :param sql: 传入的SQL 语句，参数用 %s 对应
        :param kwargs:
        :return:
        """
        return self._db.execute(sql, *kwargs)

    def query(self, sql, *kwargs):
        """
        执行 select 操作
        :param sql:
        :param kwargs:
        :return:
        """

        try:
            result = self._db.query(sql, *kwargs)
        except Exception as e:
            logger.exception(e)
            self._db.reconnect()
            result = self._db.query(sql, *kwargs)
        return result

    def get(self, sql, *kwargs):
        """
        执行 select 操作, 只返回第一个元素
        :param sql:
        :param kwargs:
        :return:
        """
        return self._db.get(sql, *kwargs)

    def reconnect(self):
        return self._db.reconnect()

    def close(self):
        return self._db.close()
