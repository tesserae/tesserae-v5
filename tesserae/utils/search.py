"""Helper class and functions for running search

Originally conceived of in order to support asynchronous web API search
"""
import multiprocessing
import queue
import time
import traceback

from bson.objectid import ObjectId

from tesserae.db import TessMongoConnection
from tesserae.db.entities import Search
import tesserae.matchers


class AsynchronousSearcher:
    """Asynchronous search resource holder

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
        results_status = Search(results_id=results_id,
                status=Search.INIT, msg='')
        connection.insert(results_status)
        try:
            search_id = results_status.id
            matcher = tesserae.matchers.matcher_map[search_type](connection)
            results_status.status = Search.RUN
            connection.update(results_status)
            text_ids, params, matches = matcher.match(search_id, **search_params)
            connection.insert_nocheck(matches)

            results_status.texts = text_ids
            results_status.parameters = params
            results_status.matches = matches
            results_status.status = Search.DONE
            results_status.msg='Done in {} seconds'.format(time.time()-start_time)
            connection.update(results_status)
        except:
            results_status.status = Search.FAILED
            results_status.msg=traceback.format_exc()
            connection.update(results_status)


def check_cache(connection, source, target, method):
    """Check whether search results are already in the database

    Parameters
    ----------
    connection : TessMongoConnection
    source
        See API documentation for form
    target
        See API documnetation for form
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
    found = [Search.json_decode(f)
        for f in connection.connection[Search.collection].find({
            'texts': [ObjectId(source['object_id']),
                ObjectId(target['object_id'])],
            'parameters.unit_types': [source['units'], target['units']],
            'parameters.method.name': method['name'],
            'parameters.method.feature': method['feature'],
            '$and': [
                {'parameters.method.stopwords': {'$all': method['stopwords']}},
                {'parameters.method.stopwords': {'$size': len(method['stopwords'])}}
            ],
            'parameters.method.freq_basis': method['freq_basis'],
            'parameters.method.max_distance': method['max_distance'],
            'parameters.method.distance_basis': method['distance_basis']
        })
    ]
    if found:
        status_found = connection.find(Search.collection,
                _id=found[0].id)
        if status_found and status_found[0].status != Search.FAILED:
            return status_found[0].results_id
    return None
