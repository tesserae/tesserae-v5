"""Database standardization for all things related to a single search.

Classes
-------
Search
    Search data model.
"""
import typing

from tesserae.db.entities import Entity


class Search(Entity):
    """The gestalt of parameters, statuses, and results of a search query

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the search. Should not be set locally.
    results_id : str, optional
        UUID for identifying search results
    search_type : str, optional
        Identification for the type of search performed
    parameters : dict, optional
        The parameters used for this search. The exact contents are fluid to
        allow for extensions to Tesserae, but this should contain all of the
        information necessary to recreate the matches.
    progress : list of dict
        A sequence of dictionaries holding both a 'stage' and a 'value' key.
        This sequence is chronologically ordered. Thus, the current progress of
        the Search can be found by looking at the last item in this list, which
        will indicate the stage of the progress and the completion of that
        stage. The value associated with 'stage' can be any string; the value
        associated with 'value' should be a real number from 0.0 to 1.0, where
        0.0 indicates no work completed in this stage and 1.0 indicating all
        work has been completed in this stage.
    status : str, optional
        Status message for determining what phase the search is in
    msg : str, optional
        Further information associated with the status
    """

    collection = 'searches'

    INIT = 'Initialized'
    RUN = 'Running'
    DONE = 'Done'
    FAILED = 'Failed'

    def __init__(
        self, id=None, results_id=None, search_type=None, parameters=None,
            progress=None, status=None, msg=None, matches=None):
        super().__init__(id=id)
        self.results_id: typing.Optional[str] = results_id \
            if results_id is not None else ''
        self.search_type: typing.Optional[str] = search_type \
            if search_type is not None else ''
        self.parameters: typing.Mapping[typing.Any, typing.Any] = parameters \
            if parameters is not None else {}
        self.progress: typing.List[typing.Any] = progress \
            if progress is not None \
            else [{'stage': 'initialize', 'value': 0.0}]
        self.status: typing.Optional[str] = status \
            if status is not None else Search.FAILED
        self.msg: typing.Optional[str] = msg \
            if msg is not None else ''

    def unique_values(self):
        uniques = {
            'results_id': self.results_id
        }
        return uniques

    def __repr__(self):
        return (
            f'Search(results_id={self.results_id}, '
            f'search_type={self.search_type}, '
            f'parameters={self.parameters}, '
            f'status={self.status}, msg={self.msg})'
        )

    def update_current_stage_value(self, value):
        self.progress[-1]['value'] = value

    def add_new_stage(self, stage):
        self.progress.append({'stage': stage, 'value': 0.0})
