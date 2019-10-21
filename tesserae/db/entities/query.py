"""Database standardization for completed search.

Classes
-------
Query
    Search query data model
"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities import Entity


class Query(Entity):
    """Search parameters associated with results and uuid.

    The Query object associates search queries with uuids that are used to
    retrieve search results for the API.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    results_id : str, optional
        uuid associated with this query.
    matchset_id : bson.objectid.ObjectId, optional
        The database id of the MatchSet created for this search.
    parameters : dict, optional
        The parameters for the search. The exact contents are fluid to allow
        for extensions to Tesserae, but this should contain all of the
        information necessary to rerun the query.

    Attributes
    ----------
    id : bson.objectid.ObjectId
        Database id of the text. Should not be set locally.
    results_id : str
        uuid used to identify query submitted.
    matchset_id : bson.objectid.ObjectId
        Database id of MatchSet created for this search.
    parameters : dict
        The parameters submitted for search.
    """

    collection = 'queries'

    def __init__(self, id=None, results_id=None, matchset_id=None,
                 parameters=None):
        super(Query, self).__init__(id=id)
        self.results_id: typing.Optional[str] = results_id
        self.matchset_id: typing.Optional[ObjectId] = matchset_id
        self.parameters: typing.Dict = \
            parameters if parameters is not None else {}

    def unique_values(self):
        uniques = {
            'results_id': self.results_id,
            'matchset_id': self.matchset_id,
            'parameters': self.parameters
        }
        return uniques

    def __repr__(self):
        return (
            f'Query(results_id={self.results_id}, '
            f'matchset_id={self.matchset_id}, parameters={self.parameters})'
        )
