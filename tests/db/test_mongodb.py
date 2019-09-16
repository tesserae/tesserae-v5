"""Unit tests for the MongoDB connection, operations, and sanitizers.

Fixtures
--------
populate
    Populate the database with test entries.
depopulate
    Remove all test entries from the database.


Functions
---------
test_init
    Test schemes for initializing TessMongoConnection.
test_insert
test_find
test_update
test_delete
    Test CRUD operations mediated by TessMongoConnection.
test_create_filter
    Test the sanitizer for creating MongoDB filters.

"""

import pytest
import random
import string

from tesserae.db.mongodb import TessMongoConnection, to_query_list, \
                                to_query_range

import datetime

import pymongo
import pymongo.results

from tesserae.db.entities import entity_map

def gen_chars():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(10))

@pytest.fixture(scope='module')
def depopulate(connection):
    """Clean up the test database.

    Fixtures
    --------
    connection
        Provides a connection to the test database.
    """
    # Stall until the tests run
    yield None

    # Once tests are finished, delete all entries from the database.
    for key, val in entity_map.items():
        collection = entity_map[key].collection
        connection[collection].delete_many({})


@pytest.fixture(scope='module')
def populate(connection, test_data):
    """Insert test entries into the database.

    Fixtures
    --------
    connection
        Provides a connection to the test database.
    test_data
        Loads the test data for database interaction evaluation.
    """
    for key, val in test_data.items():
        collection = entity_map[key].collection
        result = connection[collection].insert_many(val)
        for entry in val:
            entry['id'] = entry['_id']
            del entry['_id']

    yield test_data

    # for key in test_data.keys():
    #     connection[key].delete_many({})


def test_init(request):
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


def test_create_filter(request):
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


def test_insert(request, test_data, depopulate):
    conf = request.config
    conn = TessMongoConnection(conf.getoption('db_host'),
                               conf.getoption('db_port'),
                               conf.getoption('db_user'),
                               password=conf.getoption('db_passwd',
                                                       default=None),
                               db=conf.getoption('db_name',
                                                 default=None))

    for key, val in test_data.items():
        entity = entity_map[key]
        ents = [entity(**entry) for entry in val]

        # Insert the result for the first time. Should return the standard
        # pymongo.collection.insert_many result.
        result = conn.insert(ents)

        assert isinstance(result, pymongo.results.InsertManyResult)
        for i in range(len(result.inserted_ids)):
            assert ents[i].id == result.inserted_ids[i]

        # Attempt to insert the same entities a second time. Should return an
        # empty list.
        result = conn.insert(ents)
        assert result == []

        # Clean up
        conn.connection[key].delete_many({'_id': {'$in': [e.id for e in ents]}})

def test_find(request, populate):
    conf = request.config
    conn = TessMongoConnection(conf.getoption('db_host'),
                               conf.getoption('db_port'),
                               conf.getoption('db_user'),
                               password=conf.getoption('db_passwd',
                                                       default=None),
                               db=conf.getoption('db_name',
                                                 default=None))

    for key, val in populate.items():
        # Test finding every entry in the database.
        result = conn.find(key)
        assert len(result) == len(val)
        for r in result:
            match = [all([getattr(r, k) == val[i][k] for k in val[i].keys()])
                     for i in range(len(val))]
            assert sum([1 if m else 0 for m in match]) == 1

        # Test finding individual entries
        for entry in val:
            result = conn.find(key, _id=entry['id'])
            assert len(result) == 1
            assert all([getattr(result[0], k) == entry[k]
                        for k in entry.keys()])

        # Test finding non-existent entries
        result = conn.find(key, foo='bar')
        assert result == []


def test_update(request, populate):
    conf = request.config
    conn = TessMongoConnection(conf.getoption('db_host'),
                               conf.getoption('db_port'),
                               conf.getoption('db_user'),
                               password=conf.getoption('db_passwd',
                                                       default=None),
                               db=conf.getoption('db_name',
                                                 default=None))

    for key, val in populate.items():
        entity = entity_map[key]

        # test updating one entity at a time
        guinea_pig = entity(**val[0])
        guinea_pig.eats = 'pellet'
        result = conn.update(guinea_pig)
        assert result.matched_count == 1
        assert result.modified_count == 1
        assert conn.connection[key].count_documents({'_id': guinea_pig.id}) == 1
        found = conn.connection[key].find({'_id': guinea_pig.id})
        assert found[0]['eats'] == 'pellet'

        # test updating multiple entities at a time
        ents = [entity(**entry) for entry in val]
        changes = { e.id : ( gen_chars(), gen_chars() ) for e in ents }
        for e in ents:
            k, v = changes[e.id]
            setattr(e, k, v)
        result = conn.update(ents)
        assert result.matched_count == len(val)
        assert result.modified_count == len(val)
        found = conn.connection[key].find()
        assert all([
            changes[doc['_id']][0] in doc and
            doc[changes[doc['_id']][0]] == changes[doc['_id']][1]
            for doc in found])


def test_delete(request, populate):
    conf = request.config
    conn = TessMongoConnection(conf.getoption('db_host'),
                               conf.getoption('db_port'),
                               conf.getoption('db_user'),
                               password=conf.getoption('db_passwd',
                                                       default=None),
                               db=conf.getoption('db_name',
                                                 default=None))
    print(conn.connection.name)

    for key, val in populate.items():
        entity = entity_map[key]
        ents = [entity(**entry) for entry in val]

        # Test deleting one entry
        result = conn.delete([ents[0]])
        assert result.deleted_count == 1

        assert conn.connection[key].count_documents({'_id': ents[0].id}) == 0
        found = conn.connection[key].find()
        print(list(found))
        assert conn.connection[key].count_documents({}) == len(val) - 1

        # Test deleting the collection's entries
        result = conn.delete(ents)
        assert result.deleted_count == len(val) - 1

        # Ensure that the collection is empty
        found = conn.connection[key].find()
        assert conn.connection[key].count_documents({}) == 0


def test_to_query_list():
    assert to_query_list(1) == [1]
    assert to_query_list(1.0) == [1.0]
    assert to_query_list('foo') == ['foo']
    assert to_query_list(True) == [True]
    assert to_query_list(False) == [False]
    assert to_query_list(None) == [None]
    assert to_query_list((1, 2, 3)) == [1, 2, 3]
    assert to_query_list(['foo', 'bar', 'baz']) == ['foo', 'bar', 'baz']


def test_to_query_range():
    assert to_query_range(0, 1) == (0, 1)
    assert to_query_range(1, 0) == (0, 1)
    assert to_query_range(-100, 1) == (-100, 1)
    assert to_query_range(0, -111) == (-111, 0)
