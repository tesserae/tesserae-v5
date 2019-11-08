"""Database standardization for text matches.

Classes
-------
ResultsPair
    Pair of MatchSet ID with search results ID assigned by API
"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities.entity import Entity
from tesserae.db.entities.unit import Unit
from tesserae.db.entities.token import Token


class ResultsPair(Entity):
    """Matching data between texts.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    match_set_id : bson.objectid.ObjectId
        Match set corresponding to the results_id
    results_id : str
        uuid corresponding to the match set produced by a search

    """

    collection = 'results_pairs'

    def __init__(self, id=None, match_set_id=None, results_id=None):
        super(ResultsPair, self).__init__(id=id)
        self.match_set_id: typing.Optional[ObjectId] = match_set_id
        self.results_id: typing.Optional[str] = results_id

    def unique_values(self):
        uniques = {
            'match_set_id': self.match_set_id,
            'results_id': self.results_id
        }
        return uniques

