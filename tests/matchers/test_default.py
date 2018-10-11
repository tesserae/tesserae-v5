import pytest

from tesserae.matchers import DefaultMatcher

import re
import os

import numpy as np
import pymongo

from tesserae.db import TessMongoConnection, Match, Text, Token


@pytest.fixture(scope='module')
def reference_matches(tessfiles):
    matches = []
    for root, dirs, files in os.walk(tessfiles):
        for fname in files:
            if fname.find('tesresults') >= 0:
                metadata = {}
                matchset = []

                with open(os.path.join(root, fname), 'r') as f:
                    for line in f.readlines():
                        if re.search(r'^[\d]+.+[\d]+$', line, flags=re.UNICODE):
                            match = {}
                            parts = line.split('\t')
                            match['result'] = int(parts[0])
                            match['target_loc'] = parts[1].strip('"')
                            match['target_text'] = parts[2].strip('"')
                            match['source_loc'] = parts[3].strip('"')
                            match['source_text'] = parts[4].strip('"')
                            match['shared'] = \
                                re.split(r'; ', parts[5].strip('"'), flags=re.UNICODE)
                            match['score'] = int(parts[6])
                            matchset.append(match)
                        elif re.search(r'^[#].+[=].+$', line, flags=re.UNICODE):
                            parts = line.split()
                            if re.search(r'^[\d]+$', parts[3], flags=re.UNICODE):
                                parts[3] = int(parts[3])
                            metadata[parts[1]] = parts[3]
                matches.append((metadata, matchset))
    return matches


class TestDefaultMatcher(object):
    def test_init(self, request):
        # Test creating a TessMongoConnection for the test database without
        # database name
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None))
        m = DefaultMatcher(conn)
        assert isinstance(m.connection.connection, pymongo.database.Database)
        assert m.connection.connection.client.address == \
            (conf.getoption('db_host'), conf.getoption('db_port'))
        assert m.connection.connection.name == 'tesserae'
        assert m.matches == []

        # Test getting a MongoClient for the test database with database name
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db=conf.getoption('db_name',
                                                     default=None))
        m = DefaultMatcher(conn)
        assert isinstance(m.connection.connection, pymongo.database.Database)
        assert m.connection.connection.client.address == \
            (conf.getoption('db_host'), conf.getoption('db_port'))
        assert m.connection.connection.name == 'tess_test'
        assert m.matches == []

        # Test getting a MongoClient for the test database with database name
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db='foobar')
        m = DefaultMatcher(conn)
        assert isinstance(m.connection.connection, pymongo.database.Database)
        assert m.connection.connection.client.address == \
            (conf.getoption('db_host'), conf.getoption('db_port'))
        assert m.connection.connection.name == 'foobar'
        assert m.matches == []

    def test_clear(self):
        m = DefaultMatcher(None)

        m.clear()
        assert m.matches == []

        items = [None, Match(), 1, 1.0, 'a', True, False, ['foo'], (1,)]
        for i in range(len(items)):
            m.matches.append(items[i])
            m.clear()
            assert m.matches == []

            m.matches.extend(items[:i + 1])
            m.clear()
            assert m.matches == []

    def test_frequency_distance(self):
        m = DefaultMatcher(None)

        test_vals = [
            np.array([[[1, 1], [1, 2]], [[1, 1], [1, 2]]]),
            np.array([[[1, 2], [1, 1]], [[1, 1], [1, 2]]]),
            np.array([[[1, 1], [1, 2]], [[1, 2], [1, 1]]]),
            np.array([[[1, 2], [1, 1]], [[1, 2], [1, 1]]]),
            np.array([[[1, 1], [3, 2]], [[1, 1], [3, 2]]]),
            np.array([[[1, 2], [3, 1]], [[1, 1], [3, 2]]]),
            np.array([[[1, 1], [3, 2]], [[1, 2], [3, 1]]]),
            np.array([[[1, 2], [3, 1]], [[1, 2], [3, 1]]]),
        ]

        answers = [
            np.array([2, 2]),
            np.array([2, 2]),
            np.array([2, 2]),
            np.array([2, 2]),
            np.array([2, 2]),
            np.array([2, 2]),
            np.array([2, 2]),
            np.array([2, 2]),
        ]

        for i, val in enumerate(test_vals):
            assert np.all(m.frequency_distance(val) == answers[i])



    def test_match(self, request, populate, reference_matches):
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db=conf.getoption('db_name',
                                                     default=None))
        for t in populate['texts']:
            start = -1
            if t['language'] == 'latin':
                start = t['path'].find('la/')
            if t['language'] == 'greek':
                start = t['path'].find('grc/')
            if start > 0:
                t['path'] = t['path'][start:]

                                   
        m = DefaultMatcher(conn)
        for ref in reference_matches:
            metadata = ref[0]
            correct = ref[1]
            source = [t for t in populate['texts'] if re.search(metadata['source'], t['path'])]
            target = [t for t in populate['texts'] if re.search(metadata['target'], t['path'])]
            texts = [Text.json_decode(source[0]), Text.json_decode(target[0])]

            matches = m.match(texts, metadata['unit'], metadata['feature'],
                              stopwords=metadata['stopsize'],
                              stopword_basis=metadata['stbasis'],
                              score_basis=metadata['scorebase'],
                              frequency_basis=metadata['freqbasis'],
                              max_distance=metadata['max_dist'],
                              distance_metric=metadata['dibasis'])
            print(matches)
            assert len(matches) == len(correct)

    def test_retrieve_frequencies(self, request, populate):
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db=conf.getoption('db_name',
                                                     default=None))
        m = DefaultMatcher(conn)

        for text in populate['texts']:
            text = Text.json_decode(text)
            
            start = -1
            if text.language == 'latin':
                start = text.path.find('la/')
            if text.language == 'greek':
                start = text.path.find('grc/')
            if start >= 0:
                text.path = text.path[start:]

            tokens = [t for t in populate['tokens'] if t['text'] == text.path]
            correct = [f for f in populate['frequencies']
                       if f['text'] == text.path]
            frequencies, _ = m.retrieve_frequencies([text], tokens, [text])
            assert len(frequencies) > 0
            assert len(frequencies) == len(correct)
            for c in correct:
                assert c['form'] in frequencies

    def test_retrieve_tokens(self, request, populate):
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db=conf.getoption('db_name',
                                                     default=None))
        m = DefaultMatcher(conn)
        
        for text in populate['texts']:
            text = Text.json_decode(text)
            
            start = -1
            if text.language == 'latin':
                start = text.path.find('la/')
            if text.language == 'greek':
                start = text.path.find('grc/')
            if start >= 0:
                text.path = text.path[start:]

            correct = [t for t in populate['tokens'] if t['text'] == text.path]
            correct.sort(key=lambda x: x['index'])
            tokens = m.retrieve_tokens([text])
            assert len(tokens) > 0
            assert len(tokens[0]) == len(correct)
            for t in tokens[0]:
                assert t.json_encode() == correct[t.index]

    def test_retrieve_units(self, request, populate):
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db=conf.getoption('db_name',
                                                     default=None))
        m = DefaultMatcher(conn)
        for text in populate['texts']:
            text = Text.json_decode(text)
            
            start = -1
            if text.language == 'latin':
                start = text.path.find('la/')
            if text.language == 'greek':
                start = text.path.find('grc/')
            if start >= 0:
                text.path = text.path[start:]

            correct = [u for u in populate['units']
                      if u['text'] == text.path and u['unit_type'] == 'line']
            correct.sort(key=lambda x: x['index'])
            units = m.retrieve_units([text], 'line')
            assert len(units[0]) > 0
            assert len(units[0]) == len(correct)
            for u in units[0]:
                assert u.json_encode() == correct[u.index]

    def test_span_distance(self):
        m = DefaultMatcher(None)

        large_array = np.arange(1000)

        for i in range(2, 100):
            # Test with a basic two-index pair
            index_vector = [[1, i] for _ in range(i)]
            dists = [i for _ in range(i)]
            assert np.all(m.span_distance(index_vector) == dists)

            # Test with a larger list of indices
            index_vector = [list(range(1, i + 1)) for j in range(i)]
            dists = [i for _ in range(i)]
            assert np.all(m.span_distance(index_vector) == dists)

            # Test with two indices in a random order
            index_vector = [np.random.permutation(large_array)
                            for _ in range(i)]
            dists = np.max(index_vector, axis=-1) - \
                np.min(index_vector, axis=-1) + 1
            assert np.all(m.span_distance(index_vector) == dists)

            # Test with a large array of randomly ordered indices
            index_vector = [np.random.permutation(large_array)
                            for _ in range(i)]
            dists = np.max(index_vector, axis=-1) - \
                np.min(index_vector, axis=-1) + 1
            assert np.all(m.span_distance(index_vector) == dists)

        with pytest.raises(ValueError):
            index_vector = [[1, 1]]
            m.span_distance(index_vector)

        with pytest.raises(ValueError):
            index_vector = [[1, 1], [4, 8]]
            m.span_distance(index_vector)

        with pytest.raises(ValueError):
            index_vector = [[37, 21], [1, 1]]
            m.span_distance(index_vector)

        with pytest.raises(ValueError):
            index_vector = [[37, 21], [1, 1], [4, 8]]
            m.span_distance(index_vector)
