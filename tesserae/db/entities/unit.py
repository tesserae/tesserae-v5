"""Database standardization for text units.

Classes
-------
Unit
    Text unit data model with token indices.
"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities.entity import Entity


class Unit(Entity):
    """Group of words that make up a set to match on.

    Units are the chunks of text that matches are computed on. Units can
    come in the flavor of lines in a poem, sentences, paragraphs, etc.

    Parameters
    ----------
    id : bson.ObjectId, optional
        Database id of the text. Should not be set locally.
    text : str, optional
        The text that contains this unit.
    index : int, optional
        The order of this unit in the text. This is relative to Units of a
        particular type.
    unit_type : str, optional
        How the chunk of text in this Unit was defined, e.g., "line",
        "phrase", etc.
    tokens : list of tesserae.db.Token or bson.objectid.ObjectId, optional
        The tokens that make up this unit.

    Attributes
    ----------
    id : bson.ObjectId
        Database id of the text. Should not be set locally.
    text : str
        The text that contains this unit.
    index : int
        The order of this unit in the text. This is relative to Units of a
        particular type.
    tags : list of str
        The in-text locale tag(s) associated with the unit. Correponds to,
        e.g., lines of a poem or sentences/paragraphs of prose.
    unit_type : str
        How the chunk of text in this Unit was defined, e.g., "line",
        "phrase", etc.
    tokens : list of tesserae.db.Token or bson.objectid.ObjectId
        The tokens that make up this unit.

    """

    collection = 'units'

    def __init__(self, id=None, text=None, index=None, tags=None, unit_type=None,
                 tokens=None):
        super(Unit, self).__init__(id=id)
        self.text: typing.Optional[ObjectId] = text
        self.index: typing.Optional[int] = index
        self.tags: typing.List[str] = tags if tags is not None else []
        self.unit_type: typing.Optional[str] = unit_type
        self.tokens: typing.List[typing.Union[ObjectId, Entity]] = \
            tokens if tokens is not None else []

    def json_encode(self, exclude=None):
        self._ignore = [self.text, self.tokens]
        self.text = self.text.id
        self.tokens = [t.id for t in self.tokens]

        obj = super(Unit, self).json_encode(exclude=exclude)

        self.text, self.tokens = self._ignore
        del self._ignore

        return obj

    def unique_values(self):
        uniques = {
            'text': self.text.id,
            'index': self.index,
            'unit_type': self.unit_type}
        return uniques
