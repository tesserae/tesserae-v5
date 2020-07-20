"""Job coordination code"""
import multiprocessing
import queue

from tesserae.db import TessMongoConnection


class JobQueue:
    """Resource holder for Tesserae operations

    Attributes
    ----------
    workers : list of JobWorker
        the workers this object has created
    queue : multiprocessing.Queue
        work queue which workers listen on

    """

    def __init__(self, num_workers, db_cred):
        """Store parameters to be used in initializing resources

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
            cur_proc = JobWorker(self.db_cred, self.queue)
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
            self.queue.put((None, None))
        for worker in self.workers:
            worker.join()

    def queue_job(self, instructions, kwargs):
        """Queues job for processing

        Parameters
        ----------
        instructions : (TessMongoConnection, ...) -> None
            a function that takes a database connection as its first argument
            and any number of named arguments after
        kwargs : dict
            named values to provide to ``instructions``

        """
        self.queue.put_nowait((instructions, kwargs))


class JobWorker(multiprocessing.Process):
    """Worker process waiting for job to execute

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
        """Waits for job"""
        connection = TessMongoConnection(**db_cred)
        while True:
            instructions, kwargs = queue.get(block=True)
            if instructions is None:
                break
            instructions(connection, **kwargs)
