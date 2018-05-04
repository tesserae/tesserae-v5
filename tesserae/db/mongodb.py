"""Functions for interacting with MongoDB.

"""
try:
    # Python 3.x
    from urllib.parse import quote_plus
except ImportError:
    # Python 2.x
    from urllib import quote_plus

import pymongo


def get_client(host, port, user, pwd=None, db=None):
    """Connect to a MongoDB instance.

    Standardizes database access to abstract away `pymongo` boilerplate code.

    Parameters
    ----------
    host : str
        Name of the machine hosting the MongoDB instance.
    port : int
        The port that MongoDB is listening on.
    user : str
        The user to log in as.
    pwd : str
        The password for ``user``.
    db : str
        The database to access.

    Returns
    -------
    client : `pymongo.MongoClient`
    """
    # Set up the MongoClient
    client = pymongo.MongoClient(host=host, port=port, username=user, password=pwd)

    # If a specific database is requested, access that database
    if db is not None:
        client = client[db]

    return client


def create_filter(**kwargs):
    """Create a filter for querying a MongoDB collection.


    """
    f = {}

    if len(kwargs) == 1:
        k, v = list(kwargs.items())[0]
        f[k] = {'$in': v, '$exists': True}
    elif len(kwargs) > 1:
        f['$and'] = []
        for k, v in kwargs.items():
            if v is not None:
                f['$and'].append({k: {'$exists': True, '$in': v}})

    return f
