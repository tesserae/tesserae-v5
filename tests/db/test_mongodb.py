import pytest

from tesserae.db.mongodb import get_connection, create_filter

import datetime

import pymongo


def test_init(request):
    """Unit tests for `tesserae.db.mongodb.TesseraeMongoDB`

    Parameters
    ----------
    request : pytest builtin fixture
        Metadata about the pytest instance.
    """
    # Test getting a MongoClient for the test database without database name
    conf = request.config
    conn = get_connection(conf.getoption('db_host'),
                          conf.getoption('db_port'),
                          conf.getoption('db_user'),
                          password=conf.getoption('db_passwd', default=None))
    assert isinstance(conn, pymongo.database.Database)
    assert conn.client.address == (conf.getoption('db_host'),
                                     conf.getoption('db_port'))
    assert conn.name == 'tesserae'

    # Test getting a MongoClient for the test database with database name
    conf = request.config
    conn = get_connection(conf.getoption('db_host'),
                          conf.getoption('db_port'),
                          conf.getoption('db_user'),
                          password=conf.getoption('db_passwd', default=None),
                          db=conf.getoption('db_name', default=None))
    assert isinstance(conn, pymongo.database.Database)
    assert conn.client.address == (conf.getoption('db_host'),
                                   conf.getoption('db_port'))
    assert conn.name == 'tess_test'

    # Test getting a MongoClient for the test database with database name
    conf = request.config
    conn = get_connection(conf.getoption('db_host'),
                          conf.getoption('db_port'),
                          conf.getoption('db_user'),
                          password=conf.getoption('db_passwd', default=None),
                          db='foobar')
    assert isinstance(conn, pymongo.database.Database)
    assert conn.client.address == (conf.getoption('db_host'),
                                   conf.getoption('db_port'))
    assert conn.name == 'foobar'


def test_create_filter():
    # Test with no filters applied
    f = create_filter()
    assert f == {}

    # Test with a single argument
    f = create_filter(foo='bar')
    assert f == {'foo': {'$in': ['bar'], '$exists': True}}

    # Test with a single negated argument
    f = create_filter(foo_not='bar')
    assert f == {'foo': {'$nin': ['bar'], '$exists': True}}

    # Test with a single argument in list form
    f = create_filter(foo=['bar'])
    assert f == {'foo': {'$in': ['bar'], '$exists': True}}

    # Test with a single argument in list form
    f = create_filter(foo_not=['bar'])
    assert f == {'foo': {'$in': ['bar'], '$exists': True}}

    # Test with a single argument in list form
    f = create_filter(foo_not=['bar', 'baz'])
    assert f == {'foo': {'$nin': ['bar', 'baz'], '$exists': True}}

    # Test with a single integer argument
    f = create_filter(foo=1)
    assert f == {'foo': {'$gte': 1, '$lte': 1, '$exists': True}}

    # Test with a single negated integer argument
    f = create_filter(foo_not=1)
    assert f == {'foo': {'$lt': 1, '$gt': 1, '$exists': True}}

    # Test with a single float argument
    f = create_filter(foo=1.0)
    assert f == {'foo': {'$gte': 1.0, '$lte': 1.0, '$exists': True}}

    # Test with a single negated float argument
    f = create_filter(foo_not=1.0)
    assert f == {'foo': {'$lt': 1.0, '$gt': 1.0, '$exists': True}}

    # Test with a single datetime argument
    f = create_filter(foo=datetime.datetime(1970, 1, 1))
    assert f == {'foo': {'$gte': datetime.datetime(1970, 1, 1),
                         '$lte': datetime.datetime(1970, 1, 1),
                         '$exists': True}}

    # Test with a single negated datetime argument
    f = create_filter(foo_not=datetime.datetime(1970, 1, 1))
    assert f == {'foo': {'$lt': datetime.datetime(1970, 1, 1),
                         '$gt': datetime.datetime(1970, 1, 1),
                         '$exists': True}}

    # Test with a single integer argument
    f = create_filter(foo=(1, 10))
    assert f == {'foo': {'$gte': 1, '$lte': 10, '$exists': True}}

    # Test with a single negated integer argument
    f = create_filter(foo_not=1)
    assert f == {'foo': {'$lt': 1, '$gt': 10, '$exists': True}}

    # Test with a single float argument
    f = create_filter(foo=(1.0, 37.3409))
    assert f == {'foo': {'$gte': 1.0, '$lte': 37.3409, '$exists': True}}

    # Test with a single negated float argument
    f = create_filter(foo_not=(1.0, 37.3409))
    assert f == {'foo': {'$lt': 1.0, '$gt': 37.3409, '$exists': True}}

    # Test with a single datetime argument
    f = create_filter(foo=(datetime.datetime(1970, 1, 1),
                           datetime.datetime(1984, 1, 1)))
    assert f == {'foo': {'$gte': datetime.datetime(1970, 1, 1),
                         '$lte': datetime.datetime(1984, 1, 1),
                         '$exists': True}}

    # Test with a single negated datetime argument
    f = create_filter(foo_not=(datetime.datetime(1970, 1, 1),
                               datetime.datetime(1984, 1, 1)))
    assert f == {'foo': {'$lt': datetime.datetime(1970, 1, 1),
                         '$gt': datetime.datetime(1984, 1, 1),
                         '$exists': True}}


def test_to_query_list():
    pass


def test_to_query_datetime():
    pass


def test_to_query_range():
    pass
