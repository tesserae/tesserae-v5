import pytest

import json
import glob
import os

from tesserae.db.mongodb import TessMongoConnection


@pytest.fixture(scope='session')
def token_connection(request):
    conf = request.config
    conn = TessMongoConnection(conf.getoption('db_host'),
                               conf.getoption('db_port'),
                               conf.getoption('db_user'),
                               password=conf.getoption('db_passwd',
                                                       default=None),
                               db=conf.getoption('db_name',
                                                 default=None))
    return conn


def file_grabber(base):
    token_files = []
    for root, dirs, files in os.walk(base):
        if len(files) > 0:
            for f in files:
                _, ext = os.path.splitext(f)
                if ext == '.tess':
                    token_files.append(os.path.join(root, f))
    return token_files


@pytest.fixture(scope='session')
def latin_files(tessfiles):
    return file_grabber(os.path.join(tessfiles, 'la'))


@pytest.fixture(scope='session')
def greek_files(tessfiles):
    return file_grabber(os.path.join(tessfiles, 'grc'))
