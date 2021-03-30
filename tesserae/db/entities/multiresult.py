"""Database standardization for multitext results

Classes
-------
MultiResult
    Multitext search result associated with a particular Match
"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities.entity import Entity


class MultiResult(Entity):
    """Multitext information for a Match

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    search_id : bson.objectid.ObjectId, optional
        Database id of search to which this match belongs.
    match_id : bson.objectid.ObjectId, optional
        Database id of Match to which this MultiResult belongs
    bigram : list of str, optional
        String representations of two of the matched features from the
        associated Match
    units : list of bson.objectid.ObjectId, optional
        Database id of Units looked at during multitext search and that contain
        the associated bigram; positions within this list correspond with
        positions in the ``scores`` list
    scores : list of float, optional
        Scores for each of the Units in ``units``; these are calculated using
        the bigram as the matched words, the frequency information taken from
        the Text from which the Unit came, and the distance information taken
        from the Unit; positions within this list correspond with positions in
        the ``units`` list

    """

    collection = 'multiresults'

    def __init__(self, id=None, search_id=None, match_id=None, bigram=None,
                 units=None, scores=None):
        super().__init__(id=id)
        self.search_id: typing.Optiona[ObjectId] = search_id
        self.match_id: typing.Optional[ObjectId] = match_id
        self.bigram: typing.Optional[typing.List[str]] = bigram \
            if bigram is not None else []
        self.units: typing.Optional[typing.List[ObjectId]] = units \
            if units is not None else []
        self.scores: typing.Optional[typing.List[float]] = scores \
            if scores is not None else []

    def unique_values(self):
        uniques = {
            'search_id': self.search_id,
            'match_id': self.match_id,
            'bigram': self.bigram
        }
        return uniques

    def __repr__(self):
        return (
            f'MultiResult(search_id={self.search_id}, '
            f'match_id={self.match_id}, bigram={self.bigram}, '
            f'units={self.units}, scores={self.scores})'
        )
