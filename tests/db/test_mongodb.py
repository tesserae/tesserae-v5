import pytest

from tesserae.db.mongodb import get_client, create_filter

import pymongo


def test_get_client(request):
    """Unit tests for `tesserae.db.mongodb.get_client`

    Parameters
    ----------
    request : pytest builtin feature
        Metadata about the pytest instance.
    """
    # Test getting a MongoClient for the test database without database name
    conf = request.config
    client = get_client(conf.getoption('db_host'),
                        conf.getoption('db_port'),
                        conf.getoption('db_user'),
                        pwd=conf.getoption('db_passwd', default=None))
    assert isinstance(client, pymongo.MongoClient)
    assert client.address == (conf.getoption('db_host'),
                              conf.getoption('db_port'))

    # Test getting a MongoClient for the test database with database name
    client = get_client(conf.getoption('db_host'),
                        conf.getoption('db_port'),
                        conf.getoption('db_user'),
                        pwd=conf.getoption('db_passwd', default=None),
                        db=conf.getoption('db_name', default=None))
    assert isinstance(client, pymongo.database.Database)
    assert client._Database__client.address == (conf.getoption('db_host'),
                                                conf.getoption('db_port'))
    assert client._Database__name == conf.getoption('db_name', default=None)


def test_create_filter():
    # Test with no filters applied
    f = create_filter()
    assert f == {}

    # Test with a single argument
    f = create_filter(foo=['bar'])
    assert f == {'foo': {'$in': ['bar'], '$exists': True}}
