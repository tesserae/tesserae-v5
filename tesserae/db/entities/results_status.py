"""Database standardization for text matches.

Classes
-------
ResultsStatus
    Information regarding search results for a specific ID assigned by the API

"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities.entity import Entity
from tesserae.db.entities.unit import Unit
from tesserae.db.entities.token import Token


class ResultsStatus(Entity):
    """Matching data between texts.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    results_id : str
        UUID for identifying search results
    status : str
        Status message for determining what phase the results are in
    match_set_id : bson.objectid.ObjectId, optional
        Match set corresponding to the search results for results_id
    msg : str, optional
        Further information associated with the status

    """

    collection = 'results_statuses'

    INIT = 'Initialized'
    DONE = 'Done'
    FAILED = 'Failed'

    def __init__(self, id=None, results_id=None, status=None,
            match_set_id=None, msg=None):
        super().__init__(id=id)
        self.results_id: str = results_id
        self.status : str = status
        self.match_set_id: typing.Optional[ObjectId] = match_set_id
        self.msg : typing.Optional[str] = msg

    def unique_values(self):
        uniques = {
            'results_id': self.results_id
        }
        return uniques

