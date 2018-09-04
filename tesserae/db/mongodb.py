"""Functions for interacting with MongoDB.

The functions defined in this module are utilities that standardize
interactions with MongoDB, including connecting, creating query filters, and
santizing data.

Routine Listings
----------------
get_connection
    Connect to a Tesserae MongoDB instance.
create_filter
    Create a filter for querying a MongoDB collection.
to_query_list
    Convert a sequence to a list for a query filter.
to_query_range
    Prepare a pair of numeric values for range-based queries.

Notes
-----
All functions defined in this module that directly interact with the database
use the `pymongo`_ library.

.. _pymongo: https://api.mongodb.com/python/current/

"""

import datetime
from typing import Iterable
try:
    # Python 3.x
    from urllib.parse import quote_plus
except ImportError:
    # Python 2.x
    from urllib import quote_plus

import pymongo


def get_connection(host, port, user, password=None, db='tesserae', **kwargs):
    """Connect to a Tesserae db instance.

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
    connection : `pymongo.database.Database`
        A connection to the Tesserae database.
    """
    conn = pymongo.MongoClient(host=host, port=port, username=user,
                               password=password, **kwargs)
    conn = conn[db]
    return conn


def create_filter(**kwargs):
    """Create a filter for querying a MongoDB collection.

    This is a utility function for creating MongoDB query filters from
    arbitrary key/value pairs.

    Parameters
    ----------
    **kwargs
        Key/value pairs where key is the feature to filter on and value
        contains the filter values.

    Returns
    -------
    filter : dict
        A dictionary organized to filter MongoDB documents by ``**kwargs``. If
        no key/value pairs were provided, return an empty dictionary.

    Notes
    -----
    The filters returned by this function use `MongoDB Query Operators`_ to
    define compound conditions, such as ``$and``, ``$in``, ``$nin``, etc.

     .. _MongoDB Query Operators: https://docs.mongodb.com/manual/reference/operator/query/

    Examples
    --------
    >>> create_filter()
    {}

    Passing a single key/value pair creates a filter that will retrieve
    documents for which the filter exists and the value in the document matches
    one of the values in the filter.

    >>> create_filter(language='latin')
    {"language": {"$in": ["latin"], "$exists": True}}
    >>> create_filter(language=['latin', 'greek'])
    {"language": {"$in": ["latin", "greek"], "$exists": True}}

    Passing multiple key/value pairs creates a filter with a compound condition
    over multiple fields in a document. Returned documents will match on at
    least one value associated with each key.

    >>> create_filter(language=['latin', 'greek'], unit_types=['line'])
    {"$and": [{"language": {"$in": ["latin", "greek"], "$exists": True}},
              {"unit_types": {"$in": ["line"], "$exists": True}}]}

    Ranges of numeric or datetime values may be searched by passing a length-2
    tuple with the lower and upper bounds of the range.

    >>> create_filter(year=(19, 29))
    {"year": {"$gte": 19, "$lte": 29, "$exists": True}}

    Adding a ``_not`` to the end of a key will filter create a filter that
    excludes documents where the field is equal to one of the supplied values.

    >>> create_filter(language_not="greek")
    {"language": {"$nin": ["greek"], "$exists": True}}
    >>> create_filter(language=["latin", "english"], language_not="greek")
    {"language": {"$in": ["latin", "english"], "$nin": ["greek"],
                  "$exists: True"}}

    For numeric or datetime values, the ``_not`` format creates a filter that
    only includes results from outside of the given range.

    >>> create_filter(year_not=(19, 29))
    {"year": {"$lt": 19, "$gt": 29, "$exists": True}}
    """
    f = {}

    # Convert the values to a standard form.
    for k, v in kwargs.items():
        if v is not None:
            exclude = False
            if k.find('_not') >= 0:
                k = k.split('_')[0]
                exclude = True

            if k not in f:
                f[k] = {'$exists': True}

            q = {}

            if isinstance(v, (int, float, datetime.datetime)):
                v = to_query_range(v, v)
                if not exclude:
                    q = {'$gte': v[0], '$lte': v[1]}
                else:
                    q = {'$lt': v[0], '$gt': v[1]}
            elif isinstance(v, tuple) and len(v) == 2:
                v = to_query_range(*v)
                if not exclude:
                    q = {'$gte': v[0], '$lte': v[1]}
                else:
                    q = {'$lt': v[0], '$gt': v[1]}
            else:
                v = to_query_list(v)
                if not exclude:
                    q = {'$in': v}
                else:
                    q = {'$nin': v}
            f[k].update(q)

    if len(f) > 1:
        f = {'$and': [{k: v} for k, v in f.items()]}

    return f


def to_query_list(item):
    if isinstance(item, str) or not isinstance(item, Iterable):
        item = [item]
    else:
        item = list(item)
    return item


def to_query_range(lower, upper):
    converted = sorted([lower, upper])
    return tuple(converted)
