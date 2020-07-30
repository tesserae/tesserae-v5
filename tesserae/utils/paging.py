"""Paging utility for easy iteration.

Functions
---------
pages
    Iterate over pages of results.
"""
import math

from tesserae.utils.search import get_results, get_results_count, PageOptions


class Pager(PageOptions):
    def __init__(self, connection, search_id, sort_by='score', sort_order='descending', per_page=200):
        super().__init__(sort_by='score', sort_order='descending', per_page=200)
        self.connection = connection
        self.search_id = search_id
        self.results_count = get_results_count(connection, search_id)
        self.page_count = int(math.ceil(self.results_count / per_page))
        
        self._start_page = None
        self._end_page = None
        self._iter_step = None

    def __call__(self, start, end=None, step=1):
        self._start_page = start
        self._end_page = end if end else self.page_count
        self._iter_step = step
        
        return self.__iter__()

    def __getitem__(self, start, end=None, step=1):
        if not end:
            self.page_number = start
            out = get_results(self.connection, self.search_id, self)
        else:
            out = []
            for page in range(start, end, step):
                self.page_number = page
                out.append(get_results(self.connection, self.search_id, self))
        return out

    def __iter__(self):
        self.page_number = self._start_page
        return self
    
    def __next__(self):
        if self.page_number > self.end:
            raise StopIteration
        yield get_results(self.connection, self.search_id, self)
        self.page_number += self._iter_step
