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


class Match(Entity):
    """Matching data between texts.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    search_id : bson.objectid.ObjectId, optional
        Database id of search to which this match belongs.
    source_unit : bson.objectid.ObjectId or Unit, optional
        Text unit from source involved in this match
    target_unit : bson.objectid.ObjectId or Unit, optional
        Text unit from target involved in this match
    matched_features : list of str, optional
        String representations of features matched between the units
    score : float, optional
        The score of this match
    source_snippet : str, optional
        Text snippet of source unit
    target_snippet : str, optional
        Text snippet of target unit
    highlight : list of (int, int)
        Indices into unit tokens indicating which tokens matched; each item in
        the list are the pair of tokens that matched; the first in the pair
        corresponds to the source unit token index, the second to the target
        unit token index

    """

    collection = 'matches'

    def __init__(
            self, id=None, search_id=None, source_unit=None,
            target_unit=None, source_tag='source',
            target_tag='target', matched_features=None, score=None,
            source_snippet='', target_snippet='', highlight=None):
        super(Match, self).__init__(id=id)
        self.search_id: typing.Optional[ObjectId] = search_id
        self.source_unit: typing.Optional[typing.Union[ObjectId, Unit]] = \
            source_unit
        self.target_unit: typing.Optional[typing.Union[ObjectId, Unit]] = \
            target_unit
        self.source_tag: typing.Optional[str] = source_tag
        self.target_tag: typing.Optional[str] = target_tag
        self.matched_features: typing.Optional[typing.List[str]] = \
            matched_features if matched_features is not None else []
        self.score: typing.Optional[float] = score
        self.source_snippet: typing.Optional[str] = source_snippet
        self.target_snippet: typing.Optional[str] = target_snippet
        self.highlight: typing.List[typing.Tuple[int, int]] = \
            highlight

    def json_encode(self, exclude=None):
        self._ignore = [self.source_unit, self.target_unit]
        if isinstance(self.source_unit, Entity):
            self.source_unit = self.source_unit.id
        if isinstance(self.target_unit, Entity):
            self.target_unit = self.target_unit.id

        obj = super(Match, self).json_encode(exclude=exclude)

        self.source_unit, self.target_unit = self._ignore
        del self._ignore

        return obj

    def unique_values(self):
        uniques = {
            'search_id': self.search_id,
            'source_unit': self.source_unit,
            'target_unit': self.target_unit
        }
        return uniques

    def __repr__(self):
        return (
            f'Match(search_id={self.search_id}, '
            f'source_unit={self.source_unit}, target_unit={self.target_unit}, '
            f'source_tag={self.source_tag}, target_tag={self.target_tag}, '
            f'matched_features={self.matched_features}, '
            f'score={self.score}, '
            f'source_snippet={self.source_snippet}, '
            f'target_snippet={self.target_snippet}, '
            f'highlight={self.highlight})'
        )
