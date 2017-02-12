# -*- coding:utf-8 -*-

from pymongo import MongoClient
from src.config import mongo_article_config
from signal import *
import sys
import atexit

common_settings = {
    "serverSelectionTimeoutMS": 10 * 1000,  # 初次建立连接的时候, 默认连接时长, 如果 mongo挂了 这里就是最大连接时长, ms为单位
    "connectTimeoutMS": 10 * 1000,  # 心跳检测中,最长响应时间
    # "socketTimeoutMS":10*1000,# 已经建立了的连接, 执行一次查询最大等候时间
    "maxPoolSize": 100,
    "minPoolSize": 0,
    "waitQueueMultiple": 10,
}

mongo_config = mongo_article_config
# client_to_phonebook_data = MongoClient(mongo_config.mongodb_address_mobile)
client_to_phonebook_new_data = MongoClient(mongo_config.mongodb_address_phonebook, **common_settings)


def init_new_collection(collection_name):
    """
    初始化到某一个具体的 Collection
    :param collection_name:
    :return:
    """
    if collection_name == "test_features":
        db_address = mongo_config.mongodb_address_test_features
        dbname = mongo_config.db_name_mobile_test_features
        user_name = mongo_config.user_mobile_test_features
        passwd = mongo_config.passwd_mobile_test_features
    else:
        print('wrong collection name')
        db_address = ''
        dbname = ''
        user_name = ''
        passwd = ''

    client = MongoClient(db_address, **common_settings)
    db = getattr(client, dbname)
    db.authenticate(user_name, passwd)
    return getattr(db, collection_name)


class MongodbTestFeatures(object):
    def __init__(self):
        self.o2o_features = init_new_collection("test_features")

    def return_conn(self):
        return self.o2o_features


"""
以下代码，保证 Mongo在程序结束后能正常关闭
"""


def close_db():
    pass


def clean(*args):
    print("closing DB by signals from outside")
    close_db()
    sys.exit(0)


for sig in (SIGABRT, SIGILL, SIGINT, SIGSEGV, SIGTERM):
    signal(sig, clean)


def exit_handler():
    print('closing Mongo DB after any excuation')
    close_db()


atexit.register(exit_handler)
