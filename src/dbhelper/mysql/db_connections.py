# coding=utf-8

import torndb
from src.config import *
from .mysql_helper import MySQLHelper
from utils import get_now


def sql_connection_torndb(mysql_config):
    return torndb.Connection(
        host=mysql_config.host, database=mysql_config.db_name,
        user=mysql_config.user, password=mysql_config.passwd
    )

db_record_object = sql_connection_torndb(mysql_record_config)
try:
    db_record_real_object = sql_connection_torndb(mysql_record_real_config)
except:
    db_record_real_object = sql_connection_torndb(mysql_record_config)

# db_time = sql_connection_torndb(mysql_feature_record_config)

db_record = MySQLHelper(db_record_object)
db_record_real = MySQLHelper(db_record_real_object)
db_time = None
print("create mysql connection")
"""
以下代码，保证 MySQL在程序结束后能正常关闭
"""


def close_db():
    db_record.close()


from signal import *
import sys


def clean(*args):
    print("closing DB by signals from outside")
    close_db()
    sys.exit(0)

for sig in (SIGABRT, SIGILL, SIGINT, SIGSEGV, SIGTERM):
    signal(sig, clean)

import atexit


def exit_handler():
    print('closing DB after any excuation using dbhelper')
    close_db()
atexit.register(exit_handler)

if __name__ == '__main__':
    sql = '''
    select *
    from check_answers
    limit 1
    '''
    db_result = db_record.query(sql)
    for line in db_result:
        print(line)
    pass