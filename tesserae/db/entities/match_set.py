"""Database standardization for token features.

Classes
-------
FeatureSet
    Token features data model.
"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities import Entity


class MatchSet(Entity):
    """Collection of matches along with the parameters of the match.

    Matching in Tesserae is parameterized in a number of dimensions, and the
    definition of these parameters should be fluid. The MatchSet organizes sets
    of Match entities by their match parameters, allowing for one point from
    which to collect matches when querying the database.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    matches : list of bson.objectid.ObjectId, optional
        The database ids of the the matches in the set. These ids should be
        generated by the MongoDB instance itself on insert.
    parameters : dict, optional
        The parameters of the match. The exact contents are fluid to allow for
        extensions to Tesserae, but this should contain all of the information
        necessary to recreate the matches.

    Attributes
    ----------
    id : bson.objectid.ObjectId
        Database id of the text. Should not be set locally.
    matches : list of bson.objectid.ObjectId or tesserae.db.entities.Match
        Match entities belonging to this set
    parameters : dict
        The parameters of the match. The exact contents are fluid to allow for
        extensions to Tesserae, but this should contain all of the information
        necessary to recreate the matches.
    """

    collection = 'match_sets'

    def __init__(self, id=None, texts=None, unit_types=None, feature=None,
                 parameters=None, matches=None):
        super(MatchSet, self).__init__(id=id)
        self.texts: typing.List[typing.Union[ObjectId, Entity]] = \
            texts if texts is not None else []
        self.unit_types: typing.Optional[typing.List[str]] = unit_types
        self.feature: typing.Optional[str] = feature
        self.parameters: typing.Dict = \
            parameters if parameters is not None else {}
        self.matches = matches if matches is not None else []

    def json_encode(self, exclude=None):
        self._ignore = [self.texts, self.matches]
        self.texts = [t.id if isinstance(t, Entity) else t for t in self.texts]
        self.matches = [m.id if isinstance(m, Entity) else m
                for m in self.matches]

        obj = super(MatchSet, self).json_encode(exclude=exclude)

        self.texts, self.matches = self._ignore
        del self._ignore

        return obj

    def unique_values(self):
        uniques = {
            'texts': [t.id if isinstance(t, Entity) else t
                      for t in self.texts],
            'unit_types': self.unit_types,
            'feature': self.feature,
            'parameters': self.parameters,
            'matches': [m.id if isinstance(m, Entity) else m
                for m in self.matches]
        }
        return uniques
