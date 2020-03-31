"""Helper functions for running Tesserae search"""
from collections import defaultdict
import itertools
import time
import traceback

from tesserae.db.entities import Feature, Property, Search, Unit
import tesserae.matchers


def submit_search(jobqueue, results_id, search_type, search_params):
    """Submit a job for Tesserae search

    Parameters
    ----------
    jobqueue : tesserae.utils.coordinate.JobQueue
    results_id : str
        UUID to associate with search to be performed
    search_type : str
        the search to perform; must be a key in tesserae.matchers.matcher_map
    search_params : dict
        parameter names mapped to arguments to be used for the search

    """
    kwargs = {
        'results_id': results_id,
        'search_type': search_type,
        'search_params': search_params
    }
    jobqueue.queue_job(_run_search, kwargs)


def _run_search(connection, results_id, search_type, search_params):
    """Instructions for running Tesserae search

    Parameters
    ----------
    connection : TessMongoConnection
    results_id : str
        UUID to associate with search to be performed
    search_type : str
        the search to perform; must be a key in tesserae.matchers.matcher_map
    search_params : dict
        parameter names mapped to arguments to be used for the search

    """
    start_time = time.time()
    parameters = {
        'source': {
            'object_id': str(search_params['source'].text.id),
            'units': search_params['source'].unit_type
        },
        'target': {
            'object_id': str(search_params['target'].text.id),
            'units': search_params['target'].unit_type
        },
        'method': {
            'name': search_type,
            'feature': search_params['feature'],
            'stopwords': search_params['stopwords'],
            'freq_basis': search_params['freq_basis'],
            'max_distance': search_params['max_distance'],
            'distance_basis': search_params['distance_basis']
        }
    }
    results_status = Search(
        results_id=results_id,
        status=Search.INIT, msg='',
        parameters=parameters
    )
    connection.insert(results_status)
    try:
        search_id = results_status.id
        matcher = tesserae.matchers.matcher_map[search_type](connection)
        results_status.status = Search.RUN
        connection.update(results_status)
        matches = matcher.match(search_id, **search_params)
        connection.insert_nocheck(matches)

        results_status.status = Search.DONE
        results_status.msg = 'Done in {} seconds'.format(
            time.time() - start_time)
        connection.update(results_status)
    # we want to catch all errors and log them into the Search entity
    except:  # noqa: E722
        results_status.status = Search.FAILED
        results_status.msg = traceback.format_exc()
        connection.update(results_status)


def check_cache(connection, source, target, method):
    """Check whether search results are already in the database

    Parameters
    ----------
    connection : TessMongoConnection
    source
        See API documentation for form
    target
        See API documentation for form
    method
        See API documentation for form

    Returns
    -------
    UUID or None
        If the search results are already in the database, return the
        results_id associated with them; otherwise return None

    Notes
    -----
    Helpful links
        https://docs.mongodb.com/manual/tutorial/query-embedded-documents/
        https://docs.mongodb.com/manual/tutorial/query-arrays/
        https://docs.mongodb.com/manual/reference/operator/query/and/
    """
    found = [
        Search.json_decode(f)
        for f in connection.connection[Search.collection].find({
            'parameters.source.object_id': str(source['object_id']),
            'parameters.source.units': source['units'],
            'parameters.target.object_id': str(target['object_id']),
            'parameters.target.units': target['units'],
            'parameters.method.name': method['name'],
            'parameters.method.feature': method['feature'],
            '$and': [
                {'parameters.method.stopwords': {'$all': method['stopwords']}},
                {'parameters.method.stopwords': {
                    '$size': len(method['stopwords'])}}
            ],
            'parameters.method.freq_basis': method['freq_basis'],
            'parameters.method.max_distance': method['max_distance'],
            'parameters.method.distance_basis': method['distance_basis']
        })
    ]
    if found:
        status_found = connection.find(
            Search.collection,
            _id=found[0].id)
        if status_found and status_found[0].status != Search.FAILED:
            return status_found[0].results_id
    return None
