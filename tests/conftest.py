"""Utility operations for unit tests across multiple modules.
"""
import pytest

import datetime
import getpass
import glob
import json
import os

import numpy as np
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
def connection(request):
    conf = request.config
    client = pymongo.MongoClient(host=conf.getoption('db_host'),
                                 port=conf.getoption('db_port'),
                                 username=conf.getoption('db_user',
                                                         default=None),
                                 password=conf.getoption('db_passwd',
                                                         default=None))
    return client[conf.getoption('db_name')]


@pytest.fixture(scope='session')
def populate(connection, tessfiles):
    # Load in test entries from tests/test_db_entries.json
    directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(directory, 'test_db_entries.json'), 'r') as f:
        test_entries = json.load(f)

    for text in test_entries['texts']:
        text['path'] = os.path.join(tessfiles, text['path'])

    # Insert all of the docs
    for collection, docs in test_entries.items():
        if len(docs) > 0:
            connection[collection].insert_many(docs)

    yield test_entries

    # Clean up the test database for a clean slate next time
    for collection in test_entries:
        connection[collection].delete_many({})


@pytest.fixture(scope='session')
def tessfiles():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'tessfiles'))
