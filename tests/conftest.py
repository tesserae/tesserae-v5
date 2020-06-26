"""Utility operations for unit tests across multiple modules.
"""
from pathlib import Path
import getpass
import json
import os
import pprint
import tempfile

import pymongo
import pytest

from tesserae.db import TessMongoConnection
from tesserae.db.entities import Text
from tesserae.utils import ingest_text
from tesserae.utils.delete import obliterate
from tesserae.utils.multitext import BigramWriter
from tesserae.utils.search import get_results


# Make sure that bigram databases are written out to a temporary location
BigramWriter.BIGRAM_DB_DIR = tempfile.mkdtemp()


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
def mini_latin_metadata(tessfiles_latin_path):
    return [
        {
            'title': 'miniaeneid',
            'author': 'minivergil',
            'language': 'latin',
            'year': -19,
            'path': str(tessfiles_latin_path.joinpath('mini.aen.tess')),
            'is_prose': False
        },
        {
            'title': 'miniphar',
            'author': 'minilucan',
            'language': 'latin',
            'year': 65,
            'path': str(tessfiles_latin_path.joinpath('mini.phar.tess')),
            'is_prose': False
        },
    ]


@pytest.fixture(scope='session')
def mini_greek_metadata(tessfiles_greek_path):
    return [
        {
            'title': 'miniiliad',
            'author': 'minihomer',
            'language': 'greek',
            'year': -1260,
            'path': str(tessfiles_greek_path.joinpath('mini.il.tess')),
            'is_prose': False
        },
        {
            'title': 'minigorgis',
            'author': 'miniplato',
            'language': 'greek',
            'year': -283,
            'path': str(tessfiles_greek_path.joinpath('mini.gorg.tess')),
            'is_prose': True
        },
    ]


@pytest.fixture(scope='session')
def mini_punctuation_metadata(tessfiles_latin_path):
    return [
        {
            'title': 'minicivdei',
            'author': 'miniaug',
            'language': 'latin',
            'year': 426,
            'path': str(tessfiles_latin_path.joinpath('mini.aug.tess')),
            'is_prose': True
        },
        {
            'title': 'minidiv',
            'author': 'minicicero',
            'language': 'latin',
            'year': -44,
            'path': str(tessfiles_latin_path.joinpath('mini.cic.tess')),
            'is_prose': True
        },
    ]


@pytest.fixture(scope='session')
def mini_g2l_metadata(tessfiles_greek_path, tessfiles_latin_path):
    return [
        {
            'title': 'miniacharnians',
            'author': 'miniaristophanes',
            'language': 'greek',
            'year': -425,
            'path': str(tessfiles_greek_path.joinpath('mini.ach.tess')),
            'is_prose': False
        },
        {
            'title': 'minipunica',
            'author': 'minisilius',
            'language': 'latin',
            'year': 96,
            'path': str(tessfiles_latin_path.joinpath('mini.punica.tess')),
            'is_prose': False
        },
    ]


@pytest.fixture(scope='session')
def lucverg_metadata(tessfiles_latin_path):
    return [
        {
            'title': 'aeneid',
            'author': 'vergil',
            'language': 'latin',
            'year': -19,
            'path': str(tessfiles_latin_path.joinpath('vergil.aeneid.tess')),
            'is_prose': False
        },
        {
            'title': 'bellum civile',
            'author': 'lucan',
            'language': 'latin',
            'year': 65,
            'path': str(tessfiles_latin_path.joinpath(
                'lucan.bellum_civile.tess')),
            'is_prose': False
        },
    ]


@pytest.fixture(scope='session')
def tessfiles_path():
    return Path(__file__).resolve().parent.joinpath('tessfiles')


@pytest.fixture(scope='session')
def tessfiles_greek_path(tessfiles_path):
    return tessfiles_path.joinpath('grc')


@pytest.fixture(scope='session')
def tessfiles_latin_path(tessfiles_path):
    return tessfiles_path.joinpath('la')


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
def test_data(connection, tessfiles):
    # Load in test entries from tests/test_db_entries.json
    directory = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(directory, 'test_db_entries.json'), 'r') as f:
        test_entries = json.load(f)

    for text in test_entries['texts']:
        text['path'] = os.path.join(tessfiles, text['path'])

    # Insert all of the docs
    # for collection, docs in test_entries.items():
    #     if len(docs) > 0:
    #         connection[collection].insert_many(docs)

    yield test_entries

    # Clean up the test database for a clean slate next time
    for collection in test_entries:
        connection[collection].delete_many({})


@pytest.fixture(scope='session')
def tessfiles():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'tessfiles'))


@pytest.fixture(scope='session')
def minipop(request, mini_greek_metadata, mini_latin_metadata):
    conn = TessMongoConnection('localhost', 27017, None, None, 'minitess')
    for metadata in mini_greek_metadata:
        text = Text.json_decode(metadata)
        ingest_text(conn, text)
    for metadata in mini_latin_metadata:
        text = Text.json_decode(metadata)
        ingest_text(conn, text)
    yield conn
    obliterate(conn)


def _build_relations(results):
    relations = {}
    for match in results:
        target_loc = match['target_tag'].split()[-1]
        source_loc = match['source_tag'].split()[-1]
        if target_loc not in relations:
            relations[target_loc] = {source_loc: match}
        elif source_loc not in relations[target_loc]:
            relations[target_loc][source_loc] = match
    return relations


def _load_v3_results(minitext_path, tab_filename):
    tab_filepath = Path(minitext_path).resolve().parent.joinpath(tab_filename)
    v3_results = []
    with open(tab_filepath, 'r', encoding='utf-8') as ifh:
        for line in ifh:
            if not line.startswith('#'):
                break
        for line in ifh:
            # ignore headers
            # headers = line.strip().split('\t')
            break
        for line in ifh:
            data = line.strip().split('\t')
            v3_results.append({
                'source_tag': data[3][1:-1],
                'target_tag': data[1][1:-1],
                'matched_features': data[5][1:-1].split('; '),
                'score': float(data[6]),
                'source_snippet': data[4][1:-1],
                'target_snippet': data[2][1:-1],
                'highlight': ''
            })
    return v3_results


class V3Checker:
    @staticmethod
    def check_search_results(conn, results_id, textpath, tabname):
        v5_results = get_results(conn, results_id)
        v5_results.sort(key=lambda x: -x['score'])
        v3_results = _load_v3_results(textpath, tabname)
        v3_relations = _build_relations(v3_results)
        v5_relations = _build_relations(v5_results)
        score_discrepancies = []
        match_discrepancies = []
        in_v5_not_in_v3 = []
        in_v3_not_in_v5 = []
        for target_loc in v3_relations:
            for source_loc in v3_relations[target_loc]:
                if target_loc not in v5_relations or \
                        source_loc not in v5_relations[target_loc]:
                    in_v3_not_in_v5.append(
                        v3_relations[target_loc][source_loc])
                    continue
                v3_match = v3_relations[target_loc][source_loc]
                v5_match = v5_relations[target_loc][source_loc]
                v3_score = v3_match['score']
                v5_score = v5_match['score']
                if f'{v5_score:.3f}' != f'{v3_score:.3f}':
                    score_discrepancies.append((
                        target_loc, source_loc,
                        v5_score-v3_score))
                v5_match_features = set(v5_match['matched_features'])
                v3_match_features = set()
                for match_f in v3_match['matched_features']:
                    for f in match_f.split('-'):
                        v3_match_features.add(f)
                only_in_v5 = v5_match_features - v3_match_features
                only_in_v3 = v3_match_features - v5_match_features
                if only_in_v5 or only_in_v3:
                    match_discrepancies.append((
                        target_loc, source_loc, only_in_v5,
                        only_in_v3))
        for target_loc in v5_relations:
            for source_loc in v5_relations[target_loc]:
                if target_loc not in v3_relations or \
                        source_loc not in v3_relations[target_loc]:
                    in_v5_not_in_v3.append(
                        v5_relations[target_loc][source_loc])
        pprint.pprint(score_discrepancies)
        pprint.pprint(match_discrepancies)
        pprint.pprint(in_v5_not_in_v3)
        pprint.pprint(in_v3_not_in_v5)
        assert not score_discrepancies
        assert not match_discrepancies
        assert not in_v5_not_in_v3
        assert not in_v3_not_in_v5


@pytest.fixture(scope='session')
def v3checker():
    return V3Checker
