"""Helper classes for working with database entries.

These classes standardize interactions with database entries, in particular
mapping to and from JSON.

Classes
-------
Text
Unit
Token
NGram
Match

Notes
-----
Python type hints are used here as a means of controlling database entries.
MongoDB does not impose any restrictions on the data type of each field in a
document, but we like to know what to expect.

"""

import copy
import typing

from bson.objectid import ObjectId
import numpy as np


def convert_to_entity(entity_class):
    """Convert the JSON-esque output of a function to an entity.

    Parameters
    ----------
    func : function
        The function to wrap.
    entity_class : {Text,Unit,Token,NGram,Match}
        The entity class to convert the data to. This argument should be the
        class itself, not a class instance.

    Returns
    -------
    The results of ``func`` converted to an entiry object. If ``func`` returns
    a single result, this function will return a single result. If ``func``
    returns a list of results, this function will return a list of results.

    Notes
    -----
    This is a utility function for use as a decorator on functions that
    retrieve documents from a MongoDB database. This standardizes how entity
    objects are created to prevent duplicate code in other parts fo the
    Tesserae package.
    """
    if not issubclass(entity_class, Entity) or entity_class is Entity:
        raise TypeError(
            'Entity class must be one of: {}, {}, {}, {}, or {}. Supplied {}'
            .format(Text.__name__, Unit.__name__, Token.__name__,
                    NGram.__name__, Match.__name__,
                    entity_class.__class__.__name__))

    def outerwrapper(func):
        def wrapper(*args, **kwargs):
            results = func(*args, **kwargs)

            if isinstance(results, list):
                converted = []
                for r in results:
                    c = entity_class.json_decode(r)
                    converted.append(c)
            else:
                converted = entity_class.json_decode(results)

            return converted

        return wrapper
    return outerwrapper


class Entity(object):
    """Abstract database entity.

    This object represents a generic database entry (e.g., a document in
    MongoDB).
    """

    def __init__(self, id=None):
        self._id: typing.Optional[typing.Union[str, ObjectId]] = id

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def copy(self):
        attrs = self.json_encode()
        if '_id' in attrs:
            attrs['id'] = attrs['_id']
            del attrs['_id']
        return self.__class__(**attrs)

    @property
    def id(self) -> typing.Optional[typing.Union[str, ObjectId]]:
        return self._id

    def json_encode(self, exclude: typing.Optional[typing.List[str]]=None):
        """Encode this entity as a valid JSON object.

        Parameters
        ----------
        exclude : list of str, optional
            List of attributes to exclude from the result.

        Raises
        ------
        AttributeError
            Raised when a non-existent attribute is encountered in ``exclude``.
        """
        obj = copy.deepcopy(self.__dict__)
        exclude = exclude if exclude is not None else []
        for k in exclude:
            if k in obj:
                del obj[k]
        return obj

    @classmethod
    def json_decode(cls, obj: typing.Dict[str, typing.Any]):
        """Decode a JSON object to create an entity.

        Parameters
        ----------
        obj : dict
            Dictionary with key/value pairs corersponding to Entity object
            attributes.

        Returns
        -------
        entity : `tesserae.db.entities.Entity`
            Entity created from ``obj``.

        Raises
        ------
        AttributeError
            If ``obj`` contains
        """
        if '_id' in obj:
            obj['id'] = obj['_id']
            del obj['_id']
        return cls(**obj)


class Text(Entity):
    """Metadata about a text available to Tesserae.

    Text entries in the Tesserae database contain metadata about text files
    available to Tesserae. The language, title, author, and year are attributes
    of the text's creation. The id, CTS URN, hash, path, and unit types are all
    for internal bookeeping purposes.

    Parameters
    ----------
    id : str, optional
        Database id of the text. Should not be set locally.
    cts_urn : str, optional
        Uniform resource name following the Canonical Text Services convention.
        Unique identifier for the text.
    language : str, optional
        Language the text was written in.
    title : str, optional
        Title of the text.
    author : str, optional
        Full name of the text's author.
    year : int, optional
        Year that the text was written/published.
    unit_types : str or list of str, optional
        Available methods for splitting a text into units.

    Attributes
    ----------
    id : str
        Database id of the text. Should not be set locally.
    cts_urn : str
        Uniform resource name following the Canonical Text Services convention.
        Unique identifier for the text.
    language : str
        Language the text was written in.
    title : str
        Title of the text.
    author : str
        Full name of the text's author.
    year : int
        Year that the text was written/published.
    unit_types : list of str
        Available methods for splitting a text into units.

    """

    def __init__(self, id=None, cts_urn=None, language=None, title=None,
                 author=None, year=None, unit_types=None, path=None,
                 hash=None):
        super(Text, self).__init__(id=id)
        self.cts_urn: typing.Optional[str] = cts_urn
        self.language: typing.Optional[str] = language
        self.title: typing.Optional[str] = title
        self.author: typing.Optional[str] = author
        self.year: typing.Optional[int] = year
        self.unit_types: typing.List[str] = \
            unit_types if unit_types is not None else []
        self.path: typing.Optional[str] = path
        self.hash: typing.Optional[str] = hash


class Unit(Entity):
    def __init__(self, id=None, text=None, index=None, unit_type=None,
                 raw=None, tokens=None, n_grams=None):
        super(Unit, self).__init__(id=id)
        self.text: typing.Optional[str] = text
        self.index: typing.Optional[int] = index
        self.unit_type: typing.Optional[str] = unit_type
        self.raw: typing.Optional[str] = raw
        self.tokens: typing.List[typing.Union[str, ObjectId]] = \
            tokens if tokens is not None else []
        self.n_grams: typing.List[typing.Union[str, ObjectId]] = \
            n_grams if n_grams is not None else []


class Token(Entity):
    def __init__(self, id=None, language=None, raw=None, form=None,
                 lemmas=None, semantic=None, sound=None, frequencies=None):
        super(Token, self).__init__(id=id)
        self.language: typing.Optional[str] = language
        self.raw: typing.Optional[str] = raw
        self.form: typing.Optional[str] = form
        self.lemmas: typing.List[str] = lemmas if lemmas is not None else []
        self.semantic: typing.List[str] = \
            semantic if semantic is not None else []
        self.sound: typing.List[str] = sound if sound is not None else []
        self.frequencies: typing.Dict[str, int] = \
            frequencies if frequencies is not None else {}

    def __hash__(self):
        return hash(self.token_type)


class NGram(Entity):
    def __init__(self, id=None):
        super(NGram, self).__init__(id=id)
        # self.id = id

    # @property
    # def id(self) -> typing.Optional[str]:
    #     return self._attributes['id']
    #
    # @id.setter
    # def id(self, val: typing.Optional[str]):
    #     self._attributes['id'] = val


class Match(Entity):
    def __init__(self, id=None):
        super(Match, self).__init__(id=id)
        # self.id = id

    # @property
    # def id(self) -> typing.Optional[str]:
    #     return self._attributes['id']
    #
    # @id.setter
    # def id(self, val: typing.Optional[str]):
    #     self._attributes['id'] = val
