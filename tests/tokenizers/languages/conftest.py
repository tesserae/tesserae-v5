import pytest

import json
import glob
import os


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
