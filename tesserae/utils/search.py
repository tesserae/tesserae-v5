"""Helper functions for running Tesserae search"""
import datetime
import time
import traceback

from natsort import natsorted

from tesserae.db.entities import Match, Search
import tesserae.matchers

NORMAL_SEARCH = 'vanilla'


def submit_search(jobqueue, connection, results_id, search_type,
                  search_params):
    """Submit a job for Tesserae search

    Parameters
    ----------
    jobqueue : tesserae.utils.coordinate.JobQueue
    connection : TessMongoConnection
    results_id : str
        UUID to associate with search to be performed
    search_type : str
        the search to perform; must be a key in tesserae.matchers.matcher_map
    search_params : dict
        parameter names mapped to arguments to be used for the search

    """
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
            'distance_basis': search_params['distance_basis'],
            'min_score': search_params['min_score']
        }
    }
    results_status = Search(results_id=results_id,
                            search_type=NORMAL_SEARCH,
                            status=Search.INIT,
                            msg='',
                            parameters=parameters)
    connection.insert(results_status)
    kwargs = {
        'results_status': results_status,
        'search_type': search_type,
        'search_params': search_params
    }
    jobqueue.queue_job(_run_search, kwargs)


def _run_search(connection, results_status, search_type, search_params):
    """Instructions for running Tesserae search

    Parameters
    ----------
    connection : TessMongoConnection
    results_status : tesserae.db.entities.Search
        Status keeper
    search_params : dict
        parameter names mapped to arguments to be used for the search

    """
    start_time = time.time()
    try:
        matcher = tesserae.matchers.matcher_map[search_type](connection)
        results_status.update_current_stage_value(1.0)

        results_status.status = Search.RUN
        results_status.last_queried = datetime.datetime.utcnow()
        results_status.add_new_stage('match and score')
        connection.update(results_status)
        matches = matcher.match(results_status, **search_params)
        matches.sort(key=lambda m: m.score, reverse=True)
        results_status.update_current_stage_value(1.0)

        results_status.add_new_stage('save results')
        connection.update(results_status)
        stepsize = 5000
        for start in range(0, len(matches), stepsize):
            results_status.update_current_stage_value(start / len(matches))
            connection.update(results_status)
            connection.insert_nocheck(matches[start:start + stepsize])

        results_status.update_current_stage_value(1.0)
        results_status.status = Search.DONE
        results_status.msg = 'Done in {} seconds'.format(time.time() -
                                                         start_time)
        results_status.last_queried = datetime.datetime.utcnow()
        connection.update(results_status)
    # we want to catch all errors and log them into the Search entity
    except:  # noqa: E722
        results_status.status = Search.FAILED
        results_status.msg = traceback.format_exc()
        results_status.last_queried = datetime.datetime.utcnow()
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
    search_for = {
        'search_type':
        NORMAL_SEARCH,
        'parameters.source.object_id':
        str(source['object_id']),
        'parameters.source.units':
        source['units'],
        'parameters.target.object_id':
        str(target['object_id']),
        'parameters.target.units':
        target['units'],
        'parameters.method.name':
        method['name'],
        'parameters.method.feature':
        method['feature'],
        '$and': [{
            'parameters.method.stopwords': {
                '$all': method['stopwords']
            }
        }, {
            'parameters.method.stopwords': {
                '$size': len(method['stopwords'])
            }
        }],
        'parameters.method.freq_basis':
        method['freq_basis'],
        'parameters.method.max_distance':
        method['max_distance'],
        'parameters.method.distance_basis':
        method['distance_basis'],
        'parameters.method.min_score':
        method['min_score']
    }
    found = [
        Search.json_decode(f)
        for f in connection.connection[Search.collection].find(search_for)
    ]
    for s in found:
        if s.status != Search.FAILED:
            return s.results_id
    return None


class PageOptions:
    """Data structure indicating paging options for results"""
    def __init__(self,
                 sort_by=None,
                 sort_order=None,
                 per_page=None,
                 page_number=None):
        self.sort_by = sort_by
        if sort_order == 'ascending':
            self.sort_order = 1
        elif sort_order == 'descending':
            self.sort_order = -1
        else:
            self.sort_order = None
        self.per_page = int(per_page) if per_page is not None else None
        self.page_number = int(page_number) \
            if page_number is not None else None

    def all_specified(self):
        return self.sort_by is not None and \
            self.sort_order is not None and \
            self.per_page is not None and \
            self.page_number is not None


def retrieve_matches(connection, pipeline):
    """Retrieve matches as specified

    Projection is taken care of by this function

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
    pipeline : list of aggregation pipeline commands

    Returns
    -------
    list of MatchResult
    """
    final_pipeline = pipeline + [{
        '$project': {
            '_id': True,
            'source_tag': True,
            'target_tag': True,
            'matched_features': True,
            'score': True,
            'source_snippet': True,
            'target_snippet': True,
            'highlight': True
        }
    }]
    db_matches = connection.aggregate(Match.collection,
                                      final_pipeline,
                                      encode=False)
    return [{
        'object_id': str(match['_id']),
        'source_tag': match['source_tag'],
        'target_tag': match['target_tag'],
        'matched_features': match['matched_features'],
        'score': match['score'],
        'source_snippet': match['source_snippet'],
        'target_snippet': match['target_snippet'],
        'highlight': match['highlight']
    } for match in db_matches]


def retrieve_matches_by_search_id(connection, search_id):
    """Obtain all matches associated with a given Search ID

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
    search_id : ObjectId
        ObjectId of Search whose results you are trying to retrieve

    Returns
    -------
    list of MatchResult
    """
    return retrieve_matches(connection, [{'$match': {'search_id': search_id}}])


def retrieve_matches_by_page(connection, search_id, page_options):
    """Obtain relevant matches according to the paging options

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
    search_id : ObjectId
        ObjectId of Search whose results you are trying to retrieve
    page_options : PageOptions

    Returns
    -------
    list of MatchResult
    """
    all_specified = page_options.all_specified()
    if all_specified and page_options.sort_by == 'score':
        start = page_options.page_number * page_options.per_page
        return retrieve_matches(connection, [{
            '$match': {
                'search_id': search_id
            }
        }, {
            '$sort': {
                'score': page_options.sort_order
            }
        }, {
            '$skip': start
        }, {
            '$limit': page_options.per_page
        }])
    all_matches = retrieve_matches_by_search_id(connection, search_id)
    if all_specified:
        start = page_options.page_number * page_options.per_page
        end = start + page_options.per_page
        if page_options.sort_by == 'source_tag':
            all_matches = natsorted(all_matches,
                                    key=lambda x: x['source_tag'],
                                    reverse=page_options.sort_order == -1)
        if page_options.sort_by == 'target_tag':
            all_matches = natsorted(all_matches,
                                    key=lambda x: x['target_tag'],
                                    reverse=page_options.sort_order == -1)
        if page_options.sort_by == 'matched_features':
            all_matches = natsorted(all_matches,
                                    key=lambda x: x['matched_features'],
                                    reverse=page_options.sort_order == -1)
        return all_matches[start:end]
    return all_matches


def get_results(connection, search_id, page_options):
    """Retrieve search results with associated id

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
    search_id : ObjectId
        ObjectId for Search whose results you are trying to retrieve
    page_options : PageOptions

    Returns
    -------
    list of MatchResult
    """
    return retrieve_matches_by_page(connection, search_id, page_options)


def get_max_score(connection, search_id):
    """Retrieve maximum score of results with associated id

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
    search_id : ObjectId
        ObjectId of Search associated with results of interest

    Returns
    -------
    float
        Maximum score of results associated with ``search_id``
    """
    return connection.connection[Match.collection].find_one(
        {'search_id': search_id}, sort=[('score', -1)])['score']


def get_results_count(connection, search_id):
    """Retrieve maximum score of results with associated id

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
    search_id : ObjectId
        ObjectId for Search whose results you are trying to retrieve

    Returns
    -------
    float
    """
    return connection.connection[Match.collection].count_documents(
        {'search_id': search_id})
