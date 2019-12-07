import copy
import csv
import itertools
import os
from pathlib import Path
import re
import time

import math
import pytest

from tesserae.db import Feature, Match, MatchSet, Text, Token, Unit, \
                        TessMongoConnection
from tesserae.matchers.sparse_encoding import \
        SparseMatrixSearch, get_text_frequencies, get_corpus_frequencies
from tesserae.tokenizers import LatinTokenizer
from tesserae.unitizer import Unitizer
from tesserae.utils import TessFile, ingest_text
from tesserae.utils.retrieve import get_results, MatchResult


@pytest.fixture(scope='session')
def minipop(request, mini_greek_metadata, mini_latin_metadata):
    conn = TessMongoConnection('localhost', 27017, None, None, 'minitess')
    for coll_name in conn.connection.list_collection_names():
        conn.connection.drop_collection(coll_name)
    for metadata in mini_greek_metadata:
        text = Text.json_decode(metadata)
        ingest_text(conn, text)
    for metadata in mini_latin_metadata:
        text = Text.json_decode(metadata)
        ingest_text(conn, text)
    yield conn


@pytest.fixture(scope='module')
def search_connection(request):
    """Create a new TessMongoConnection for this task.

    Fixtures
    --------
    request
        The configuration to connect to the MongoDB test server.
    """
    conf = request.config
    conn = TessMongoConnection(conf.getoption('db_host'),
                               conf.getoption('db_port'),
                               conf.getoption('db_user'),
                               password=conf.getoption('db_passwd',
                                                       default=None),
                               db=conf.getoption('db_name',
                                                 default=None))
    return conn


@pytest.fixture(scope='module')
def populate_database(search_connection, test_data):
    """Set up the database to conduct searches on the test texts.

    Fixtures
    --------
    search_connection
        TessMongoConnection for search unit tests.
    test_data
        Example data for unit testing.
    """
    for text in test_data['texts']:
        tessfile = TessFile(text['path'], metadata=Text(**text))
        search_connection.insert(tessfile.metadata)
        if text['language'] == 'latin':
            tok = LatinTokenizer(search_connection)
        unitizer = Unitizer()
        tokens, tags, features = tok.tokenize(tessfile.read(), text=tessfile.metadata)
        print(features[0].json_encode())
        search_connection.update(features)
        lines, phrases = unitizer.unitize(tokens, tags, tessfile.metadata)
        search_connection.insert(lines + phrases)
        search_connection.insert(tokens)

    yield

    search_connection.connection['texts'].delete_many({})
    search_connection.connection['tokens'].delete_many({})
    #search_connection.connection['features'].delete_many({})
    search_connection.connection['units'].delete_many({})
    search_connection.connection['matches'].delete_many({})
    search_connection.connection['match_sets'].delete_many({})


@pytest.fixture(scope='module')
def search_tessfiles(search_connection, populate_database):
    """Select the texts to use in the searches.

    Fixtures
    --------
    search_connection
        TessMongoConnection for search unit tests.
    populate_database
        Set up the database to conduct searches on the test texts.
    """
    return search_connection.find('texts')


@pytest.fixture(scope='module')
def correct_results(tessfiles):
    """Tesserae v3 search results for the test texts.

    Fixtures
    --------
    tessfiles
        Path to the test .tess files.
    """
    correct_matches = []
    for root, dirs, files in os.walk(tessfiles):
        for fname in files:
            if os.path.splitext(fname)[1] == '.csv':
                results = {
                    'source': None,
                    'target': None,
                    'unit': None,
                    'feature': None,
                    'dibasis': None,
                    'matches': []
                }
                match_template = {
                    'result': None,
                    'target_locus': None,
                    'target_text': None,
                    'source_locus': None,
                    'source_text': None,
                    'shared': None,
                    'score': None
                }
                with open(os.path.join(root, fname), 'r') as f:
                    print(fname)
                    for k, line in enumerate(f.readlines()):
                        print(line)
                        if line[0] == '#' and 'source' in line:
                            start = line.find('=') + 1
                            results['source'] = line[start:].strip()
                        elif line[0] == '#' and 'target' in line:
                            start = line.find('=') + 1
                            results['target'] = line[start:].strip()
                        elif line[0] == '#' and 'unit' in line:
                            start = line.find('=') + 1
                            results['unit'] = line[start:].strip()
                        elif line[0] == '#' and 'feature' in line:
                            start = line.find('=') + 1
                            ftype = line[start:].strip()
                            results['feature'] = 'form' if ftype == 'word' else 'lemmata'
                        elif line[0] == '#' and 'dibasis' in line:
                            start = line.find('=') + 1
                            results['dibasis'] = line[start:].strip()
                        elif re.search(r'^[\d]', line[0]):
                            parts = re.split(r',"(?!,")', line)  # [p.strip('"').replace('*', '') for p in line.split(',"')]
                            parts = [p for p in parts if p]
                            print(parts)
                            this_match = copy.deepcopy(match_template)
                            this_match['result'] = int(parts[0])
                            this_match['target_locus'] = parts[1].split()[-1]
                            this_match['target_text'] = parts[2]
                            this_match['source_locus'] = parts[3].split()[-1]
                            this_match['source_text'] = parts[4]
                            this_match['shared'] = parts[5].split(',')[0].replace('-', ' ').replace(';', '').split()
                            this_match['shared'] = [s.strip('"') for s in this_match['shared']]
                            this_match['score'] = int(parts[5].split(',')[1])
                            results['matches'].append(this_match)
                correct_matches.append(results)
    return correct_matches


def lookup_entities(search_connection, match):
    units = search_connection.find(Unit.collection, _id=match.units)
    # features = search_connection.find(Feature.collection, id=match.tokens)
    match.units = units
    # match.tokens = features
    return match

def test_init(search_connection):
    engine = SparseMatrixSearch(search_connection)
    assert engine.connection is search_connection


def test_get_stoplist(search_connection):
    engine = SparseMatrixSearch(search_connection)


def test_create_stoplist(search_connection):
    engine = SparseMatrixSearch(search_connection)


def test_get_frequencies(search_connection):
    engine = SparseMatrixSearch(search_connection)


def test_match(search_connection, search_tessfiles, correct_results):
    engine = SparseMatrixSearch(search_connection)

    for result in correct_results:
        source = [t for t in search_tessfiles
                  if os.path.splitext(os.path.basename(t.path))[0] == result['source']][0]
        target = [t for t in search_tessfiles
                  if os.path.splitext(os.path.basename(t.path))[0] == result['target']][0]

        start = time.time()
        matches, ms = engine.match([source, target], result['unit'], result['feature'], stopwords=10,
                     stopword_basis='corpus', score_basis='word', distance_metric=result['dibasis'],
                     max_distance=50, min_score=6)
        print(time.time() - start)

        matches = [lookup_entities(search_connection, m) for m in matches]
        matches.sort(key=lambda x: x.score, reverse=True)

        # print(matches, result)
        # top_matches = [m for m in result['matches'] if m['score'] == 10]
        for i in range(len(matches)):
            predicted = matches[i]
            src = predicted.units[0].tags[0]
            tar = predicted.units[1].tags[0]
            correct = None

            # print(matches[i].units[0].tags, result['matches'][i]['source_locus'])
            # print(matches[i].units[0].tokens, result['matches'][i]['source_text'])
            # print(matches[i].units[1].tags, result['matches'][i]['target_locus'])
            # print(matches[i].units[1].tokens, result['matches'][i]['target_text'])
            # print([t.token for t in matches[i].tokens], result['matches'][i]['shared'])
            # print(matches[i].score, result['matches'][i]['score'])

            for m in result['matches']:
                if m['source_locus'] == src and m['target_locus'] == tar:
                    correct = m
                    break
            # print([t.token for t in predicted.tokens], correct)
            assert correct is not None, "No matching v3 result found."
            assert src == correct['source_locus']

            assert all(map(lambda x: x.token in correct['shared'], predicted.tokens))


def _load_v3_mini_text_stem_freqs(conn, metadata):
    db_cursor = conn.connection[Feature.collection].find(
            {'feature': 'form', 'language': metadata['language']},
            {'_id': False, 'index': True, 'token': True})
    token2index = {e['token']: e['index'] for e in db_cursor}
    # the .freq_score_stem file is named the same as its corresponding .tess
    # file
    freqs_path = metadata['path'][:-4] + 'freq_score_stem'
    freqs = {}
    with open(freqs_path, 'r') as ifh:
        for line in ifh:
            if line.startswith('# count:'):
                denom = int(line.split()[-1])
                break
        for line in ifh:
            line = line.strip()
            if line:
                word, count = line.split()
                freqs[token2index[word]] = float(count) / denom
    return freqs


def test_mini_text_frequencies(minipop, mini_latin_metadata,
        mini_greek_metadata):
    all_text_metadata = [m for m in itertools.chain.from_iterable(
        [mini_latin_metadata, mini_greek_metadata])]
    title2id = {t.title: t.id for t in minipop.find(
        Text.collection, title=[m['title'] for m in all_text_metadata])}
    for metadata in all_text_metadata:
        v3freqs = _load_v3_mini_text_stem_freqs(minipop, metadata)
        text_id = title2id[metadata['title']]
        v5freqs = get_text_frequencies(minipop, 'lemmata', text_id)
        for form_index, freq in v5freqs.items():
            assert form_index in v3freqs
            assert math.isclose(v3freqs[form_index], freq)


def _load_v3_results(minitext_path, tab_filename):
    tab_filepath = Path(minitext_path).resolve().parent.joinpath(tab_filename)
    v3_results = []
    with open(tab_filepath, 'r') as ifh:
        for line in ifh:
            if not line.startswith('#'):
                break
        for line in ifh:
            headers = line.strip().split('\t')
            break
        for line in ifh:
            data = line.strip().split('\t')
            v3_results.append(MatchResult(
                source=data[3][1:-1],
                target=data[1][1:-1],
                match_features=data[5][1:-1].split('; '),
                score=float(data[6]),
                source_raw=data[4][1:-1].replace('*', ''),
                target_raw=data[2][1:-1].replace('*', ''),
                highlight = ''
            ))
    return v3_results


def _check_search_results(v5_results, v3_results):
    for v5_r, v3_r in zip(v5_results, v3_results):
        assert v5_r.source.split()[-1] == v3_r.source.split()[-1]
        assert v5_r.target.split()[-1] == v3_r.target.split()[-1]
        # v3 scores were truncated to the third decimal point
        assert f'{v5_r.score:.3f}' == f'{v3_r.score:.3f}'
        v5_r_match_fs = set(v5_r.match_features)
        v3_r_match_fs = set()
        for match_f in v3_r.match_features:
            for f in match_f.split('-'):
                v3_r_match_fs.add(f)
        assert len(v3_r_match_fs) == len(v5_r_match_fs)
        assert len(v5_r_match_fs & v3_r_match_fs) == len(v5_r_match_fs)


def test_mini_latin_search_text_freqs(minipop, mini_latin_metadata):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_latin_metadata])
    matcher = SparseMatrixSearch(minipop)
    v5_matches, match_set = matcher.match(texts, 'line', 'lemmata',
            stopwords=['et', 'qui', 'quis', 'pilum', 'pila', 'signum'],
            stopword_basis='texts', score_basis='stem',
            frequency_basis='texts', max_distance=10,
            distance_metric='frequency', min_score=0)
    minipop.insert(v5_matches)
    minipop.insert(match_set)
    v5_results = get_results(minipop, match_set.id)
    v5_results = sorted(v5_results, key=lambda x: -x.score)
    v3_results = _load_v3_results(texts[0].path, 'mini_latin_results.tab')
    _check_search_results(v5_results, v3_results)


def test_mini_greek_search_text_freqs(minipop, mini_greek_metadata):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_greek_metadata])
    matcher = SparseMatrixSearch(minipop)
    v5_matches, match_set = matcher.match(texts, 'phrase', 'lemmata',
            stopwords=['ὁ', 'ὅς', 'καί', 'αβγ', 'ἐγώ', 'δέ', 'οὗτος', 'ἐμός'],
            stopword_basis='texts', score_basis='stem',
            frequency_basis='texts', max_distance=10,
            distance_metric='span', min_score=0)
    minipop.insert(v5_matches)
    minipop.insert(match_set)
    v5_results = get_results(minipop, match_set.id)
    v5_results = sorted(v5_results, key=lambda x: -x.score)
    v3_results = _load_v3_results(texts[0].path, 'mini_greek_results.tab')
    _check_search_results(v5_results, v3_results)


def _load_v3_mini_corpus_stem_freqs(conn, language, lang_path):
    lang_lookup = {'latin': 'la.mini', 'greek': 'grc.mini'}
    freqs_path = lang_path.joinpath(
            lang_lookup[language]+'.stem.freq')
    db_cursor = conn.connection[Feature.collection].find(
            {'feature': 'lemmata', 'language': language},
            {'_id': False, 'index': True, 'token': True})
    token2index = {e['token']: e['index'] for e in db_cursor}
    freqs = {}
    with open(freqs_path, 'r') as ifh:
        for line in ifh:
            if line.startswith('# count:'):
                denom = int(line.split()[-1])
                break
        for line in ifh:
            line = line.strip()
            if line:
                word, count = line.split()
                freqs[token2index[word]] = float(count) / denom
    return freqs


def test_mini_corpus_frequencies(minipop, tessfiles_greek_path,
        tessfiles_latin_path):
    for lang, lang_path in zip(['greek', 'latin'],
            [tessfiles_greek_path, tessfiles_latin_path]):
        v3freqs = _load_v3_mini_corpus_stem_freqs(minipop, lang, lang_path)
        v5freqs = get_corpus_frequencies(minipop, 'lemmata', lang)
        db_cursor = minipop.connection[Feature.collection].find(
                {'feature': 'lemmata', 'language': lang},
                {'_id': False, 'index': True, 'token': True})
        index2token = {e['index']: e['token'] for e in db_cursor}
        for f in minipop.connection[Feature.collection].find(
                {'feature': 'lemmata', 'language': lang},
                {'_id': False, 'index': True, 'token': True, 'frequencies': True}):
            print(f['token'], f['frequencies'])
        for form_index, freq in enumerate(v5freqs):
            assert form_index in v3freqs
            assert math.isclose(v3freqs[form_index], freq), \
                    f'Mismatch on {index2token[form_index]} ({form_index})'


def test_mini_latin_search_corpus_freqs(minipop, mini_latin_metadata):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_latin_metadata])
    matcher = SparseMatrixSearch(minipop)
    v5_matches, match_set = matcher.match(texts, 'line', 'lemmata',
            stopwords=6,
            stopword_basis='corpus', score_basis='stem',
            frequency_basis='corpus', max_distance=10,
            distance_metric='frequency', min_score=0)
    minipop.insert(v5_matches)
    minipop.insert(match_set)
    v5_results = get_results(minipop, match_set.id)
    v5_results = sorted(v5_results, key=lambda x: -x.score)
    v3_results = _load_v3_results(
            texts[0].path, 'mini_latin_corpus_results.tab')
    _check_search_results(v5_results, v3_results)


def test_mini_greek_search_corpus_freqs(minipop, mini_greek_metadata):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_greek_metadata])
    matcher = SparseMatrixSearch(minipop)
    v5_matches, match_set = matcher.match(texts, 'phrase', 'lemmata',
            stopwords=8,
            stopword_basis='corpus', score_basis='stem',
            frequency_basis='corpus', max_distance=10,
            distance_metric='span', min_score=0)
    minipop.insert(v5_matches)
    minipop.insert(match_set)
    v5_results = get_results(minipop, match_set.id)
    v5_results = sorted(v5_results, key=lambda x: -x.score)
    v3_results = _load_v3_results(
            texts[0].path, 'mini_greek_corpus_results.tab')
    _check_search_results(v5_results, v3_results)