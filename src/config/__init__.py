# -*- coding:utf-8 -*-
from .mongo import MongoConfig, MongoConfig2
from .mysql import MysqlConfig
from .api import ApiConfig
from .server import ServerConfig
# MySQL 配置
mysql_mobile_card_s_config = MysqlConfig("mysql_mobile_card_s")
mysql_mobile_card_m_config = MysqlConfig("mysql_mobile_card_m")
mysql_loan_core_m_config = MysqlConfig("mysql_loan_core_m")
mysql_loan_core_s_config = MysqlConfig("mysql_loan_core_s")
mysql_o2o_config = MysqlConfig("mysql_o2o")
mysql_feature_config = MysqlConfig("mysql_feature")

mysql_record_config = MysqlConfig("mysql_record")
try:
    mysql_record_real_config = MysqlConfig("mysql_record_real")
except:
    pass

mysql_feature_record_config = MysqlConfig("mysql_feature_record")

# Mongo 配置
# mongo_datacenter_config = MongoConfig("mongodb_ds")
mongo_article_config = MongoConfig2()
mongo_tb_config = MongoConfig("mongodb_tb")
mongo_phone_book_config = MongoConfig("mobile_contacts")
mongo_o2o_median_config = MongoConfig("mongodb_o2o_median_user")

api_config = ApiConfig()
server_config = ServerConfig()