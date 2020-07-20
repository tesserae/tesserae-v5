"""Database standardization for stopword lists

Classes
-------
StopwordsList
    Named list of stopwords
"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities.entity import Entity


class StopwordsList(Entity):
    """Named list of stopwords

    Parameters
    ----------
    id : bson.OjbectId, optional
        Database id of the list. Should not be set locally.
    name : str, optional
        Name of list.
    stopwords : list of str
        Stopwords included in this list.

    Attributes
    ----------
    id : bson.OjbectId
        Database id of the list. Should not be set locally.
    name : str
        Name of list.
    stopwords : list of str
        Stopwords included in this list.
    """

    collection = 'stopwords_lists'

    def __init__(self, id=None, name=None, stopwords=None):
        super(StopwordsList, self).__init__(id=id)
        self.name: typing.Optional[str] = name
        self.stopwords: typing.List[str] = \
            stopwords if stopwords is not None else []

    def __repr__(self):
        return f'StopwordsList(name={self.name}, stopwords={self.stopwords})'
