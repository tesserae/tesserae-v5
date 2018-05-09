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


def convert_to_entity(func, entity_class):
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

    def wrapper(*args, **kwargs):
        results = func(*args, **kwargs)

        if isinstance(results, list):
            converted = []
            for r in results:
                c = entity_class()
                c.json_decode(r)
                converted.append(c)
        else:
            converted = entity_class()
            converted.json_decode(results)

        return converted

    return wrapper


class Entity(object):
    """Abstract database entity.

    This object represents a generic database entry (e.g., a document in
    MongoDB).
    """

    def __init__(self):
        self._attributes = {}

    def __eq__(self, other):
        return self._attributes == other._attributes

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
        obj = copy.deepcopy(self._attributes)
        if exclude is not None:
            for k in exclude:
                if k in obj:
                    del obj[k]
                else:
                    raise AttributeError(
                        'Entity {} does not have a field {}'.format(
                            self.__class__.__name__, k))
        return obj

    def json_decode(self, obj: typing.Dict[str, typing.Any]):
        """Decode a JSON object to create an entity.


        """
        for k, v in obj.items():
            if k in self._attributes:
                self._attributes[k] = v
            else:
                raise AttributeError(
                    'Entity {} has no attribute {}. Database entry: {}'.format(
                        self.__class__.__name__, k, obj))


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
                 author=None, year=None, unit_types=None):
        super(Text, self).__init__()
        self.id = id
        self.cts_urn = cts_urn
        self.language = language
        self.title = title
        self.author = author
        self.year = year
        self.unit_types = unit_types if unit_types is not None else []

    @property
    def id(self) -> typing.Optional[str]:
        return self._attributes['id']

    @id.setter
    def id(self, val: typing.Optional[str]):
        self._attributes['id'] = val

    @property
    def cts_urn(self) -> typing.Optional[str]:
        return self._attributes['cts_urn']

    @cts_urn.setter
    def cts_urn(self, val: typing.Optional[str]) -> None:
        self._attributes['cts_urn'] = val

    @property
    def language(self) -> typing.Optional[str]:
        return self._attributes['language']

    @language.setter
    def language(self, val: typing.Optional[str]) -> None:
        self._attributes['language'] = val

    @property
    def title(self) -> typing.Optional[str]:
        return self._attributes['title']

    @title.setter
    def title(self, val: typing.Optional[str]) -> None:
        self._attributes['title'] = val

    @property
    def author(self) -> typing.Optional[str]:
        return self._attributes['author']

    @author.setter
    def author(self, val: typing.Optional[str]) -> None:
        self._attributes['author'] = val

    @property
    def year(self) -> typing.Optional[str]:
        return self._attributes['year']

    @year.setter
    def year(self, val: typing.Optional[int]) -> None:
        self._attributes['year'] = val

    @property
    def unit_types(self) -> typing.List[str]:
        return self._attributes['unit_types']

    @unit_types.setter
    def unit_types(self, val: typing.Union[str, typing.List[str]]):
        if 'unit_types' not in self._attributes:
            self._attributes['unit_types'] = []
        if isinstance(val, str):
            self._attributes['unit_types'].append(val)
        else:
            self._attributes['unit_types'] = val


class Unit(Entity):
    def __init__(self, id=None):
        super(Unit, self).__init__()
        self.id = id

    @property
    def id(self) -> typing.Optional[str]:
        return self._attributes['id']

    @id.setter
    def id(self, val: typing.Optional[str]):
        self._attributes['id'] = val


class Token(Entity):
    def __init__(self, id=None):
        super(Token, self).__init__()
        self.id = id

    @property
    def id(self) -> typing.Optional[str]:
        return self._attributes['id']

    @id.setter
    def id(self, val: typing.Optional[str]):
        self._attributes['id'] = val


class NGram(Entity):
    def __init__(self, id=None):
        super(NGram, self).__init__()
        self.id = id

    @property
    def id(self) -> typing.Optional[str]:
        return self._attributes['id']

    @id.setter
    def id(self, val: typing.Optional[str]):
        self._attributes['id'] = val


class Match(Entity):
    def __init__(self, id=None):
        super(Match, self).__init__()
        self.id = id

    @property
    def id(self) -> typing.Optional[str]:
        return self._attributes['id']

    @id.setter
    def id(self, val: typing.Optional[str]):
        self._attributes['id'] = val
