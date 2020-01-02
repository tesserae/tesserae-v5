"""Database standardization for text tokens.

Classes
-------
Token
    Text token data model with matching-related features.
"""
import collections
import copy
import typing

from bson.objectid import ObjectId

from tesserae.db.entities.entity import Entity


class Token(Entity):
    """An atomic piece of text, along with related features.

    Tokens contain the atomic pieces of text that inform Matches and make
    up Units. In addition to the raw text, the normalized form of the text
    and features like lemmata and semantic meaning are also part of a
    token.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    text : str or bson.objectid.ObjectId, optional
        The text containing this token.
    index : int, optional
        The order of this token in the text.
    display : str, optional
        The un-altered form of this token, as it appears in the original
        text.
    features : dict[str, ObjectId or list of ObjectIds]
        Mapping between feature type and Feature entities of the given type
        associated with this token


    Attributes
    ----------
    id : bson.objectid.ObjectId
        Database id of the text. Should not be set locally.
    text : str or bson.objectid.ObjectId
        The text containing this token.
    index : int
        The order of this token in the text.
    display : str
        The un-altered form of this token, as it appears in the original
        text.
    features : dict[str, ObjectId or list of ObjectIds]
        Mapping between feature type and Feature entities of the given type
        associated with this token

    """

    collection = 'tokens'

    def __init__(self, id=None, text=None, index=None, display=None,
                 features=None, line=None, phrase=None, frequency=None):
        super(Token, self).__init__(id=id)
        self.text: typing.Optional[typing.Union[Entity, ObjectId]] = text
        self.index: typing.Optional[int] = index
        self.display: typing.Optional[str] = display
        self.features: typing.Dict[
                str, typing.Union[ObjectId, typing.List[ObjectId]]] = \
            features if features is not None else {}

    def json_encode(self, exclude=None):
        self._ignore = [self.text, self.features]

        self.text = self.text.id if self.text is not None else None
        for key, val in self.features.items():
            if isinstance(val, Entity):
                self.features[key] = val.id
            elif isinstance(val, collections.Sequence):
                self.features[key] = [v.id for v in val]

        obj = super(Token, self).json_encode(exclude=exclude)

        self.text, self.features = self._ignore
        del self._ignore

        return obj

    def unique_values(self):
        uniques = {
            'text': self.text.id if isinstance(self.text, Entity) else self.text,
            'index': self.index
        }
        return uniques
