"""Database standardization for text token frequencies.

Classes
-------
Frequency
    Text token frequency data model.
"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities import Entity


class Frequency(Entity):
    """Record of the frequency with which a token appears in a text.

    The frequency is a record of how often a token as understood by its
    stem appears in a text. Tesserae uses the frequency of token normalized
    forms or token stem forms to score matches.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    text : str or bson.objectid.ObjectId, optional
        The text containing this token.
    form : str, optional
        The normalized form of this token.
    frequency : int, optional
        The number of occurrences of this token/stem in the text.

    Attributes
    ----------
    id : bson.objectid.ObjectId
        Database id of the text. Should not be set locally.
    text : str or bson.objectid.ObjectId, optional
        The text containing this token.
    form : str
        The normalized form of this token.
    frequency : int
        The number of occurrences of this token/stem in the text.

    """

    collection = 'frequencies'

    def __init__(self, id=None, text=None, feature_set=None, frequency=None):
        super(Frequency, self).__init__(id=id)
        self.text: typing.Optional[ObjectId] = text
        self.feature_set: typing.Optional[ObjectId] = feature_set
        self.frequency: typing.Optional[int] = frequency

    def json_encode(self, exclude=None):
        self._ignore = []

        if isinstance(self.text, Entity):
            self._ignore.append(self.text)
            self.text = self.text.id

        if isinstance(self.feature_set, Entity):
            self._ignore.append(self.feature_set)
            self.feature_set = self.feature_set.id

        obj = super(Frequency, self).json_encode(exclude=exclude)

        self.text, self.feature_set = self._ignore
        del self._ignore

        return obj

    def __hash__(self):
        return hash(self.feature_set)

    def unique_values(self):
        uniques = {'text': self.text.id, 'feature_set': self.feature_set.id}
        return uniques
