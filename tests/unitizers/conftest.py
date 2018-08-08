import pytest

import json
import glob
import os


def populate_units(files, unit_type):
    file_units = []
    for f in files:
        root, _ = os.path.splitext(f)
        token_file = root + '.tokens.json'
        units_file = root + '.' + unit_type + '.json'

        with open(token_file, 'r') as f:
            tokens = json.load(f)

        with open(units_file, 'r') as f:
            units = json.load(f)

        token_units = []

        for unit in units:
            raw = ''
            forms = []
            for tid in unit['TOKEN_ID']:
                raw += tokens[tid]['DISPLAY']
                if tokens[tid]['TYPE'] == 'WORD':
                    forms.append(tokens[tid]['FORM'])
            token_units.append({'raw': raw, 'forms': forms})
        file_units.append(token_units)
    return file_units


@pytest.fixture(scope='session')
def poetry_files(tessfiles):
    fpaths = []
    for root, dirs, files in os.walk(tessfiles):
        if 'new' not in root and 'poetry' in root:
            for f in files:
                if '.tess' in f:
                    fpaths.append(os.path.join(root, f))
    return fpaths


@pytest.fixture(scope='session')
def prose_files(tessfiles):
    fpaths = []
    for root, dirs, files in os.walk(tessfiles):
        if 'new' not in root and 'prose' in root:
            for f in files:
                if '.tess' in f:
                    fpaths.append(os.path.join(root, f))
    return fpaths


@pytest.fixture(scope='session')
def poetry_lines(poetry_files):
    return populate_units(poetry_files, 'lines')


@pytest.fixture(scope='session')
def poetry_phrases(poetry_files):
    return populate_units(poetry_files, 'phrases')


@pytest.fixture(scope='session')
def prose_lines(prose_files):
    return populate_units(prose_files, 'lines')


@pytest.fixture(scope='session')
def prose_phrases(prose_files):
    return populate_units(prose_files, 'phrases')
