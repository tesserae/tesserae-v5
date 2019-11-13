"""Helper class and functions for running search

Originally conceived of in order to support asynchronous web API search
"""
import multiprocessing
import queue
import time
import traceback

from tesserae.db import TessMongoConnection
from tesserae.db.entities import ResultsStatus
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
        super().__init__(target=self.await_job)
        self.connection = TessMongoConnection(**db_cred)
        self.queue = queue

    def await_job(self):
        """Waits for search job"""
        while True:
            results_id, search_type, search_params = self.queue.get(block=True)
            if results_id is None:
                break
            self.run_search(results_id, search_type, search_params)

    def run_search(self, results_id, search_type, search_params):
        """Executes search"""
        start_time = time.time()
        try:
            matcher = tesserae.matchers.search_types[search_type](self.connection)
            matches, match_set = matcher.match(**search_params)
            self.connection.insert(matches)
            self.connection.insert(match_set)
            results_status = ResultsStatus(results_id=results_id,
                    status=ResultsStatus.DONE,
                    match_set_id=match_set.id,
                    msg='Done in {} seconds'.format(time.time()-start_time))
            self.connection.insert(results_status)
        except:
            results_status = ResultsStatus(results_id=results_id,
                    status=ResultsStatus.FAILED, msg=traceback.format_exc())
            self.connection.insert(results_status)
