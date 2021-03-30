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

from collections import Iterable, Mapping
import datetime
try:
    # Python 3.x
    from urllib.parse import quote_plus
except ImportError:
    # Python 2.x
    from urllib import quote_plus
import sys

import pymongo
import six

import tesserae.db.entities


# https://goshippo.com/blog/measure-real-size-any-python-object/
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj,
                                                     (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def _extract_embedded_docs(doc):
    """Recursive function to build dot notation for MongoDB $set operation
    """
    result_keys = []
    result_vals = []
    for key, val in doc.items():
        if isinstance(val, Mapping):
            dotted_keys, dotted_vals = _extract_embedded_docs(val)
            for d_key, d_val in zip(dotted_keys, dotted_vals):
                result_keys.append(key + '.' + d_key)
                result_vals.append(d_val)
        else:
            result_keys.append(key)
            result_vals.append(val)
    return result_keys, result_vals


def _dot_notate(doc):
    """Flatten embedded documents for MongoDB $set operation
    """
    dotted_keys, dotted_vals = _extract_embedded_docs(doc)
    return {k: v for k, v in zip(dotted_keys, dotted_vals)}


class TessMongoConnection():
    """Connection to a MongoDB instance configured for Tesserae.

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

    Attributes
    ----------
    connection : `pymongo.database.Database`
        A connection to the Tesserae database.
    """
    def __init__(self, host, port, user, password, db='tesserae', **kwargs):
        conn = pymongo.MongoClient(host=host,
                                   port=port,
                                   username=user,
                                   password=password,
                                   **kwargs)
        conn = conn[db]
        self.connection = conn

    def aggregate(self, collection, pipeline, encode=True):
        """Execute a MongoDB aggregation pipeline.

        Parameters
        ----------
        collection : str
            The MongoDB collection to search.
        pipeline : list of dict
            The list of pipeline stage commands.
        encode : bool
            If True, encode the results as tesserae.db.entities.Entity
            instances. Set to False if `pipeline` will return documents that
            do not match any Entity.

        Returns
        -------
        entities : list of tesserae.db.entities.Entity or list of dict
            The documents returned from the database.
        """
        result = self.connection[collection].aggregate(pipeline,
                                                       allowDiskUse=True)
        if encode:
            entity = None
            if collection in tesserae.db.entities.entity_map:
                entity = tesserae.db.entities.entity_map[collection]

            result = [entity.json_decode(doc) for doc in result]

        return result

    def find(self, collection, sort=None, **filter_values):
        """Retrieve database entries.

        Parameters
        ----------
        collection : str
            The MongoDB collection to search.
        filter_values
            Keyword arguments with values to filter the database query.

        Returns
        -------
        entities : list of tesserae.db.entities.Entity
            The documents returned from the database.
        """
        query_filter = self.create_filter(**filter_values)
        coll = self.connection[collection]
        documents = coll.find(query_filter, sort=sort)

        entity = None
        if collection in tesserae.db.entities.entity_map:
            entity = tesserae.db.entities.entity_map[collection]

        result = [entity.json_decode(doc) for doc in documents]

        return result

    def delete(self, entity):
        """Delete one or more entries from the database.

        Parameters
        ----------
        entity : tesserae.db.entities.Entity or list of Entity
            The entities to delete from the database.

        Raises
        ------
        ValueError
            Raised when no entities could be deleted

        """
        if not isinstance(entity, list):
            entity = [entity]
        try:
            collection = self.connection[entity[0].__class__.collection]
            result = collection.delete_many(
                self.create_filter(_id=[e.id for e in entity]))
        except IndexError:
            raise ValueError("No entities provided.")
        return result

    def insert_nocheck(self, entity):
        """Insert one or more entities into the database without checking
        whether similar entities already exist in the database.  You have been
        warned.

        Parameters
        ----------
        entity : tesserae.db.entities.Entity or list of Entity
            The entities to insert into the database.

        Raises
        ------
        ValueError
            Raised when provided entity could not be inserted

        """
        if not isinstance(entity, list):
            entity = [entity]

        try:
            collection = self.connection[entity[0].__class__.collection]
            result = collection.insert_many(
                [e.json_encode(exclude=['_id']) for e in entity])
            assert len(entity) == len(result.inserted_ids)
            for e, e_id in zip(entity, result.inserted_ids):
                e.id = e_id
        except IndexError:
            result = []

        return result

    def insert(self, entity):
        """Insert one or more entities into the database.

        Parameters
        ----------
        entity : tesserae.db.entities.Entity or list of Entity
            The entities to insert into the database.

        Raises
        ------
        ValueError
            Raised when provided entity could not be inserted

        """
        if not isinstance(entity, list):
            entity = [entity]

        filter_vals = {}
        for e in entity:
            for k, v in e.unique_values().items():
                if k in filter_vals:
                    filter_vals[k].append(v)
                else:
                    filter_vals[k] = [v]

        try:
            exists = self.find(entity[0].collection, **filter_vals)
        except IndexError:
            exists = []

        if len(exists) != 0:
            exists = [e.unique_values() for e in exists]
            new_ents = []
            for e in entity:
                if e.id is None or not any(
                    [e.unique_values() == ex for ex in exists]):
                    new_ents.append(e)
            entity = new_ents

        try:
            collection = self.connection[entity[0].__class__.collection]
            result = collection.insert_many(
                [e.json_encode(exclude=['_id']) for e in entity])
            for i, e in enumerate(entity):
                e.id = result.inserted_ids[i]
        except IndexError:
            result = []

        return result

    def update(self, entity):
        """Update existing entries in the database.

        No updates are made unless all updates can be made.

        Parameters
        ----------
        entity : tesserae.db.entities.Entity or list of Entity
            The entities to update in the database.  Each entity will be found
            by its 'id', and all other attributes and corresponding values of
            the given entity will be used to update the database with a
            matching 'id'.

        Raises
        ------
        ValueError
            Raised when an entity does not have an 'id'
        pymongo.errors.BulkWriteError
            Raised when provided entities could not be updated

        Returns
        -------
        None or pymongo.results.BulkWriteResult
            If no changes were made, None is returned

        """
        if not isinstance(entity, list):
            entity = [entity]

        collection = None
        bulk = []
        no_ids = []
        for i, e in enumerate(entity):
            if e.id is not None:
                bulk.append(
                    pymongo.operations.UpdateOne(
                        {'_id': e.id},
                        {'$set': e.json_encode(exclude=['_id'])}))
            else:
                no_ids.append(i)
        if no_ids:
            raise ValueError(
                'No id for the entities at the following indices: {}'.format(
                    no_ids))
        if len(bulk) > 0:
            collection = self.connection[e.collection]
            result = collection.bulk_write(bulk)
        else:
            result = None

        return result

    def create_filter(self, **kwargs):
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
        query_filter = {}

        # Convert the values to a standard form.
        for key, val in kwargs.items():
            if val is not None:
                exclude = False
                if key.find('_not') >= 0:
                    key = key.split('_')[0]
                    exclude = True

                if key not in query_filter:
                    query_filter[key] = {'$exists': True}

                query = {}

                if isinstance(val, (int, float, datetime.datetime)):
                    val = to_query_range(val, val)
                    if not exclude:
                        query = {'$gte': val[0], '$lte': val[1]}
                    else:
                        query = {'$lt': val[0], '$gt': val[1]}
                elif isinstance(val, tuple) and len(val) == 2:
                    val = to_query_range(*val)
                    if not exclude:
                        query = {'$gte': val[0], '$lte': val[1]}
                    else:
                        query = {'$lt': val[0], '$gt': val[1]}
                else:
                    val = to_query_list(val)
                    if not exclude:
                        query = {'$in': val}
                    else:
                        query = {'$nin': val}
                query_filter[key].update(query)

        if len(query_filter) > 1:
            query_filter = {
                '$and': [{
                    key: val
                } for key, val in query_filter.items()]
            }

        return query_filter

    def to_query_list(self, item):
        """Convert value to a list for a MongoDB query.

        Parameters
        ----------
        item
            The item to convert to or place in a list.

        Returns
        -------
        converted : list
            The item placed in or converted into a list.
        """
        if isinstance(item, str) or not isinstance(item, Iterable):
            item = [item]
        else:
            item = list(item)
        return item

    def to_query_range(self, lower, upper):
        """Convert values into a range for a MongoDB query.

        Parameters
        ----------
        lower : int, float
        upper : int, float
            The bounds of the search range.

        Returns
        -------
        converted : tuple
            The bounds sorted and converted to a tuple.
        """
        converted = sorted([lower, upper])
        return tuple(converted)

    def create_indices(self):
        """Creates indices for entities for faster lookup later"""
        # index Unit entities by Text.id
        self.connection[tesserae.db.entities.Unit.collection].create_index(
            'text')
        # index Match entities by Search.id for faster search results retrieval
        self.connection[tesserae.db.entities.Match.collection].create_index([
            ('search_id', pymongo.ASCENDING),
            ('score', pymongo.DESCENDING),
        ])
        # index Search entities by uuid
        self.connection[tesserae.db.entities.Search.collection].create_index(
            'results_id')
        # index Feature entities by language and feature type
        self.connection[tesserae.db.entities.Search.collection].create_index([
            ('language', pymongo.ASCENDING),
            ('feature', pymongo.ASCENDING),
        ])
        self.connection[tesserae.db.entities.MultiResult.
                        collection].create_index('match_id')

    def drop_indices(self):
        """Drops all indices

        Might be useful if you need to rebuild indices
        """
        for coll_name in self.connection.list_collection_names():
            self.connection[coll_name].drop_indexes()


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
    conn = pymongo.MongoClient(host=host,
                               port=port,
                               username=user,
                               password=password,
                               **kwargs)
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
    if isinstance(item, six.string_types) or not isinstance(item, Iterable):
        item = [item]
    else:
        item = list(item)
    return item


def to_query_range(lower, upper):
    converted = sorted([lower, upper])
    return tuple(converted)
