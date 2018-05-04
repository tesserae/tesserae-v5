"""Utility operations for unit tests across multiple modules.
"""
import pytest

import getpass

import pymongo


def pytest_addoption(parser):
    parser.addoption('--db-host', action='store', default='127.0.0.1',
                     help='IP of the test database host')
    parser.addoption('--db-port', action='store', default=27017, type=int,
                     help='Port that the test database listens on')
    parser.addoption('--db-user', action='store',
                     help='User to log into the test database as')
    parser.addoption('--db-pwd', action='store_true',
                     help='Pass this flag to input database password on start')
    parser.addoption('--db-name', action='store', default='tess_test',
                     help='Name of the test database to use.')


def pytest_configure(config):
    if config.option.db_pwd:
        password = getpass.getpass(prompt='Test Database Password: ')
        setattr(config.option, 'db_passwd', password)


@pytest.fixture(scope='session')
def connect(request):
    conf = request.config
    client = pymongo.MongoClient(host=conf.getoption('db_host'),
                                 port=conf.getoption('db_port'),
                                 username=conf.getoption('db_user',
                                                         default=None),
                                 password=conf.getoption('db_passwd',
                                                         default=None))
    return client[conf.getoption('db_name')]
