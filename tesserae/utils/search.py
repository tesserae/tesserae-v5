"""Helper class and functions for running search

AsynchronousSearcher provides normal Tesserae search capabilities.

bigram_search enables lookup of bigrams for specified units of specified texts
"""
from collections import defaultdict
import itertools
import multiprocessing
import queue
import time
import traceback

from tesserae.db import TessMongoConnection
from tesserae.db.entities import Feature, Property, Search, Unit
import tesserae.matchers


class AsynchronousSearcher:
    """Asynchronous Tesserae search resource holder

    Attributes
    ----------
    workers : list of SearchProcess
        the workers this object has created
    queue : multiprocessing.Queue
        work queue which workers listen on

    """

    def __init__(self, num_workers, db_cred):
        """Store parameters to be used in intializing resources

        Parameters
        ----------
        num_workers : int
            number of workers to create
        db_cred : dict
            credentials to access the database; arguments should be given for
            TessMongoConnection.__init__ in kwarg unpacking format

        """
        self.num_workers = num_workers
        self.db_cred = db_cred

        self.queue = multiprocessing.Queue()
        self.workers = []
        for _ in range(self.num_workers):
            cur_proc = SearchProcess(self.db_cred, self.queue)
            cur_proc.start()
            self.workers.append(cur_proc)

    def cleanup(self, *args):
        """Clean up system resources being used by this object

        This method should be called by exit handlers in the main script

        """
        try:
            while True:
                self.queue.get_nowait()
        except queue.Empty:
            pass
        for _ in range(len(self.workers)):
            self.queue.put((None, None, None))
        for worker in self.workers:
            worker.join()

    def queue_search(self, results_id, search_type, search_params):
        """Queues search for processing

        Parameters
        ----------
        results_id : str
            UUID for identifying the search request
        search_type : str
            identifier for type of search to perform.  Available options are
            defined in tesserae.matchers.search_types (located in the
            __init__.py file).
        search_params : dict
            search parameters

        """
        self.queue.put_nowait((results_id, search_type, search_params))


class SearchProcess(multiprocessing.Process):
    """Worker process waiting for search to execute

    Listens on queue for work to do
    """

    def __init__(self, db_cred, queue):
        """Constructs a search worker

        Parameters
        ----------
        db_cred : dict
            credentials to access the database; arguments should be given for
            TessMongoConnection.__init__ in keyword format
        queue : multiprocessing.Queue
            mechanism for receiving search requests

        """
        super().__init__(target=self.await_job, args=(db_cred, queue))

    def await_job(self, db_cred, queue):
        """Waits for search job"""
        connection = TessMongoConnection(**db_cred)
        while True:
            results_id, search_type, search_params = queue.get(block=True)
            if results_id is None:
                break
            self.run_search(connection, results_id, search_type, search_params)

    def run_search(self, connection, results_id, search_type, search_params):
        """Executes search"""
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
                'freq_basis': search_params['frequency_basis'],
                'max_distance': search_params['max_distance'],
                'distance_basis': search_params['distance_metric']
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


def bigram_search(
        connection, word1_index, word2_index, feature, unit_type, text_id):
    """Retrieves all Units of a specified type containing the specified words

    Parameters
    ----------
    connection : TessMongoConnection
    word1_index, word2_index : int
        Feature index of words to be contained in a Unit
    feature : {'lemmata', 'form'}
        Feature type of words to search for
    unit_type : {'line', 'phrase'}
        Type of Units to look for
    text_id : ObjectId
        The ID of Text in whose Units the bigram is to be searched

    Returns
    -------
    list of Unit
        All Units of the specified texts and ``unit_type`` containing
        both ``word1_index`` and ``word2_index``
    """
    bigram = tuple(sorted((word1_index, word2_index)))
    bigram_data = connection.lookup_bigrams(text_id, unit_type, feature,
                                            [bigram])
    if bigram not in bigram_data:
        return []
    return connection.find(Unit.collection,
                           _id=[u for u in bigram_data[bigram]])


def multitext_search(connection, matches, feature_type, unit_type, texts):
    """Retrieves Units containing matched bigrams

    Parameters
    ----------
    connection : TessMongoConnection
    matches : list of Match
        Match entities from which matched bigrams are taken
    feature_type : {'lemmata', 'form'}
        Feature type of words to search for
    unit_type : {'line', 'phrase'}
        Type of Units to look for
    texts : list of Text
        The Texts whose Units are to be searched

    Returns
    -------
    list of dict[(str, str), list of ObjectId]
        each dictionary within the list corresponds in index to a match from
        ``matches``; the dictionary contains key-value pairs, where the key is
        a bigram and the value is a list of ObjectIds of Units of type
        ``unit_type`` that contains the bigram specified by the key; Units are
        restricted to those which are found in ``texts``
    """
    language = texts[0].language
    token2index = {
        f.token: f.index
        for f in connection.find(
            Feature.collection, feature=feature_type, language=language)}

    bigram_indices = set()
    for m in matches:
        for w1, w2 in itertools.combinations(sorted(m.matched_features), 2):
            bigram_indices.add((token2index[w1], token2index[w2]))

    bigram2units = defaultdict(list)
    for text in texts:
        bigram_data = connection.lookup_bigrams(
            text.id, unit_type, feature_type, bigram_indices)
        for bigram, data in bigram_data.items():
            print(bigram)
            print(data)
            bigram2units[bigram].extend([
                u for u in data
            ])

    return [
        {
            bigram: bigram2units[
                (token2index[bigram[0]], token2index[bigram[1]])]
            for bigram in itertools.combinations(sorted(m.matched_features), 2)
        }
        for m in matches
    ]
