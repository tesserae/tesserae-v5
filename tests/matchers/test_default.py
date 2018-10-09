import pytest

from tesserae.matchers import DefaultMatcher

import numpy as np
import pymongo

from tesserae.db import TessMongoConnection, Match


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
        pass
        # m = DefaultMatcher(None)
        #
        # frequency_vector = [1, 2]
        # assert m.frequency_distance(frequency_vector) == [1, 1]

    def test_match(self, request):
        pass

    def test_retrieve_frequencies(self, request):
        pass

    def test_retrieve_tokens(self, request):
        pass

    def test_retrieve_units(self, request):
        pass

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
