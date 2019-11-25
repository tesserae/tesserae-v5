"""Database standardization for text matches.

Classes
-------
Match
    Text match data model with token indices.
"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities.entity import Entity
from tesserae.db.entities.unit import Unit
from tesserae.db.entities.token import Token


class Match(Entity):
    """Matching data between texts.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    units : list of bson.objectid.ObjectId or list of Unit, optional
        Text units involved in this match.
    tokens : list of bson.objectid.ObjectId or list of Token, optional
        Tokens contributing to the match.
    score : float, optional
        The score of this match.

    """

    collection = 'matches'

    def __init__(self, id=None, units=None, tokens=None, score=None,
                 match_set=None):
        super(Match, self).__init__(id=id)
        self.units: typing.Optional[typing.List[ObjectId, Unit]] = \
            units if units is not None else []
        self.tokens: typing.Optional[typing.List[ObjectId, Token]] = \
            tokens if tokens is not None else []
        self.score: typing.Optional[float] = score

    def json_encode(self, exclude=None):
        self._ignore = [self.units, self.tokens]
        self.units = [u.id if isinstance(u, Entity) else u for u in self.units]
        self.tokens = [t.id if isinstance(t, Entity) else t
                       for t in self.tokens]

        obj = super(Match, self).json_encode(exclude=exclude)

        self.units, self.tokens = self._ignore
        del self._ignore

        return obj

    def unique_values(self):
        uniques = {
            'units': [u.id if isinstance(u, Entity) else u
                      for u in self.units],
            'score': self.score,
        }
        return uniques
