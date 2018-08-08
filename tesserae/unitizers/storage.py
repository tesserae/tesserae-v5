"""Utilities for interacting with units in the database.

"""
from tesserae.db import convert_to_entity, create_filter, Unit


class InvalidTokenError(Exception):
    def __init__(self, token):
        msg = 'The supplied token {} is not a valid token type or does not'
        msg += ' contain the correct fields.'
        super(InvalidTokenError, self).__init__(msg.format(token))


class DuplicateTokenError(Exception):
    """Raised when duplicate tokens exist in the database"""
    def __init__(self, token):
        msg = 'Multiple tokens {} exist in the database.'.format(token)
        super(DuplicateTokenError, self).__init__(msg)


class NoTokenError(Exception):
    """Raised when attempting to update a token that is not in the database"""
    def __init__(self, token):
        msg = 'Token {} does not exist in the database.'.format(token)
        super(NoTextError, self).__init__(msg)


@convert_to_entity(Unit)
def retrieve_unit_list(client, text=None, index=None, unit_type=None):
    """Retrieve a list of units from the database.

    Parameters
    ----------
    client : `pymongo.MongoClient`
        Client connection to the database.
    text : str or bson.objectid.ObjectId
        Filter by CTS URN or database ObjectId.
    index : int or (int, int)
        Filter by unit index in the text. Supplying a tuple will return a
        sequence of units.
    unit_type : str
        Filter by type of unit, e.g. 'line' or 'phrase'

    Returns
    -------
    units : list of tesserae.db.Unit
    """
    # Create the filter
    filter = create_filter(text=text, index=index, unit_type=unit_type)

    # Retrieve the units and put them into a nice format
    docs = client['units'].find(filter)
    units = [doc for doc in docs]
    return units


def insert_units(client, units):
    """Insert units into the database.

    Parameters
    ----------
    client : `pymongo.MongoClient`
        Client connection to the database.
    units : list of dict or list of tesserae.db.Unit
        The unit to insert. If a dict is supplied, it will be converted into a
        Unit entity for validation.
    """
    # Standardize all units as Unit entities
    entities = []
    raws = []
    for u in units:
        if not isinstance(u, Unit):
            try:
                entities.append(Unit(**u))
                raws.append(u['raw'])
            except (TypeError, KeyError, AttributeError):
                raise InvalidUnitError(u)
        else:
            entities.append(u)
            raws.append(u.raw)

    # Only insert tokens that do not already exist in the database
    db_tokens = retrieve_unit_list(client, raw=raws)
    del raws

    for u, i in enumerate(entities):
        if u.raw == db_tokens[i].raw:
            entities.remove(u)

    result = client['units'].insert_many([e.json_encode() for e in entities])
    return result
