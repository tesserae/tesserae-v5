import pytest

from tesserae.db.mongodb import TessMongoConnection

import datetime

import pymongo


class TestTessMongoConnection(object):
    def test_init(self, request):
        # Test creating a TessMongoConnection for the test database without
        # database name
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None))
        assert isinstance(conn.connection, pymongo.database.Database)
        assert conn.connection.client.address == (conf.getoption('db_host'),
                                                  conf.getoption('db_port'))
        assert conn.connection.name == 'tesserae'

        # Test getting a MongoClient for the test database with database name
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db=conf.getoption('db_name',
                                                     default=None))
        assert isinstance(conn.connection, pymongo.database.Database)
        assert conn.connection.client.address == (conf.getoption('db_host'),
                                                  conf.getoption('db_port'))
        assert conn.connection.name == 'tess_test'

        # Test getting a MongoClient for the test database with database name
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db='foobar')
        assert isinstance(conn.connection, pymongo.database.Database)
        assert conn.connection.client.address == (conf.getoption('db_host'),
                                                  conf.getoption('db_port'))
        assert conn.connection.name == 'foobar'

    def test_create_filter(self, request):
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None))

        # Test with no filters applied
        f = conn.create_filter()
        assert f == {}

        # Test with a single argument
        f = conn.create_filter(foo='bar')
        assert f == {'foo': {'$in': ['bar'], '$exists': True}}

        # Test with a single negated argument
        f = conn.create_filter(foo_not='bar')
        assert f == {'foo': {'$nin': ['bar'], '$exists': True}}

        # Test with a single argument in list form
        f = conn.create_filter(foo=['bar'])
        assert f == {'foo': {'$in': ['bar'], '$exists': True}}

        # Test with a single argument in list form
        f = conn.create_filter(foo_not=['bar'])
        assert f == {'foo': {'$nin': ['bar'], '$exists': True}}

        # Test with a single argument in list form
        f = conn.create_filter(foo=['bar', 'baz'])
        assert f == {'foo': {'$in': ['bar', 'baz'], '$exists': True}}

        # Test with a single argument in list form
        f = conn.create_filter(foo_not=['bar', 'baz'])
        assert f == {'foo': {'$nin': ['bar', 'baz'], '$exists': True}}

        # Test with a single argument in list form
        f = conn.create_filter(foo=['bar'], foo_not=['baz'])
        assert f == {'foo': {'$in': ['bar'], '$nin': ['baz'], '$exists': True}}

        # Test with a single integer argument
        f = conn.create_filter(foo=1)
        assert f == {'foo': {'$gte': 1, '$lte': 1, '$exists': True}}

        # Test with a single negated integer argument
        f = conn.create_filter(foo_not=1)
        assert f == {'foo': {'$lt': 1, '$gt': 1, '$exists': True}}

        # Test with a single float argument
        f = conn.create_filter(foo=1.0)
        assert f == {'foo': {'$gte': 1.0, '$lte': 1.0, '$exists': True}}

        # Test with a single negated float argument
        f = conn.create_filter(foo_not=1.0)
        assert f == {'foo': {'$lt': 1.0, '$gt': 1.0, '$exists': True}}

        # Test with a single datetime argument
        f = conn.create_filter(foo=datetime.datetime(1970, 1, 1))
        assert f == {'foo': {'$gte': datetime.datetime(1970, 1, 1),
                             '$lte': datetime.datetime(1970, 1, 1),
                             '$exists': True}}

        # Test with a single negated datetime argument
        f = conn.create_filter(foo_not=datetime.datetime(1970, 1, 1))
        assert f == {'foo': {'$lt': datetime.datetime(1970, 1, 1),
                             '$gt': datetime.datetime(1970, 1, 1),
                             '$exists': True}}

        # Test with a single integer argument
        f = conn.create_filter(foo=(1, 10))
        assert f == {'foo': {'$gte': 1, '$lte': 10, '$exists': True}}

        # Test with a single negated integer argument
        f = conn.create_filter(foo_not=(1, 10))
        assert f == {'foo': {'$lt': 1, '$gt': 10, '$exists': True}}

        # Test with a single float argument
        f = conn.create_filter(foo=(1.0, 37.3409))
        assert f == {'foo': {'$gte': 1.0, '$lte': 37.3409, '$exists': True}}

        # Test with a single negated float argument
        f = conn.create_filter(foo_not=(1.0, 37.3409))
        assert f == {'foo': {'$lt': 1.0, '$gt': 37.3409, '$exists': True}}

        # Test with a single datetime argument
        f = conn.create_filter(foo=(datetime.datetime(1970, 1, 1),
                               datetime.datetime(1984, 1, 1)))
        assert f == {'foo': {'$gte': datetime.datetime(1970, 1, 1),
                             '$lte': datetime.datetime(1984, 1, 1),
                             '$exists': True}}

        # Test with a single negated datetime argument
        f = conn.create_filter(foo_not=(datetime.datetime(1970, 1, 1),
                                   datetime.datetime(1984, 1, 1)))
        assert f == {'foo': {'$lt': datetime.datetime(1970, 1, 1),
                             '$gt': datetime.datetime(1984, 1, 1),
                             '$exists': True}}

    def test_delete(self, request, populate):
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db=conf.getoption('db_name',
                                                     default=None))

        for entity_type in populate:
            for query_ent in populate[entity_type]:
                res = conn.delete(entity_type, _id=query_ent['_id'])
                assert res.deleted_count == 1
                assert res.raw_result == query_ent
                conn.insert(entity_type, **query_ent)

                conn.delete(entity_type, **query_ent)
                assert res.deleted_count == 1
                assert res.raw_result == query_ent
                conn.insert(entity_type, **query_ent)

    def test_find(self, request, populate):
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db=conf.getoption('db_name',
                                                     default=None))
        for entity_type in populate:
            for query_ent in populate[entity_type]:
                ents = conn.find(entity_type, _id=query_ent['_id'])
                assert ents[0].json_encode == query_ent

                ents = conn.find(entity_type, **query_ent)
                assert ents[0].json_encode == query_ent

        for text in populate['texts']:
            ents = conn.find('tokens', text=text.path)
            correct_tokens = [t for t in populate['tokens']
                              if t['text'] == text.path]
            assert len(ents) == len(correct_tokens)

            ents = conn.find('frequencies', text=text.path)
            correct_tokens = [t for t in populate['frequencies']
                              if t['text'] == text.path]
            assert len(ents) == len(correct_tokens)

            ents = conn.find('units', text=text.path)
            correct_tokens = [t for t in populate['units']
                              if t['text'] == text.path]
            assert len(ents) == len(correct_tokens)


    def test_insert(self, request, populate):
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db=conf.getoption('db_name',
                                                     default=None))
        for entity_type in populate:
            conn.delete(entity_type, _id=[t['_id'] for t in
                                          populate[entity_type]])
            for query_ent in populate[entity_type]:
                del query_ent['_id']
                res = conn.insert(entity_type, **query_ent)
                assert res.inserted_ids[0] == query_ent['_id']

    def test_to_query_list(self):
        pass

    def test_to_query_range(self):
        pass

    def test_update(self, request, populate):
        conf = request.config
        conn = TessMongoConnection(conf.getoption('db_host'),
                                   conf.getoption('db_port'),
                                   conf.getoption('db_user'),
                                   password=conf.getoption('db_passwd',
                                                           default=None),
                                   db=conf.getoption('db_name',
                                                     default=None))
        for entity_type in populate:
            for query_ent in populate[entity_type]:
                query_ent['foo'] = 'bar'
                res = conn.insert(entity_type, **query_ent)
                assert res.matched_count == 1
                assert res.modified_count == 1
                assert res.raw_result == query_ent
