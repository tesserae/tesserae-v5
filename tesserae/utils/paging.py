"""Paging utility for easy iteration.

Classes
-------
Pager
    Iterate over pages of results in the database.
"""
import math

from tesserae.utils.search import get_results, get_results_count, PageOptions


class Pager(PageOptions):
    """Iterate over pages of results in the database.

    Parameters
    ----------
    connection : `tesserae.db.mongodb.TessMongoConection`
        Connection to the MongoDB instance.
    search_id : str or `bson.objectid.ObjectID`
        The database ID of the search to paginate.
    sort_by : str
        The document field on which to sort pages. Default: ``'score'``.
    sort_order : {'descending','ascending'}
        The order in which to sort along ``sort_by``.
        Default: ``'descending'``.
    per_page : int
        The number of results to include per page. Default: ``200``.

    Attributes
    ----------
    connection : `tesserae.db.mongodb.TessMongoConection`
        Connection to the MongoDB instance.
    search_id : str or `bson.objectid.ObjectID`
        The database ID of the search to paginate.
    results_count : int
        The total number of results in the given search.
    page_count : int
        The number of pages given ``self.results_count`` and
        ``self.per_page``.
    """
    def __init__(self,
                 connection,
                 search_id,
                 sort_by='score',
                 sort_order='descending',
                 per_page=200):
        super().__init__(sort_by='score',
                         sort_order='descending',
                         per_page=200,
                         page_number=0)
        self.connection = connection
        self.search_id = search_id
        self.results_count = get_results_count(connection, search_id)
        self.page_count = int(math.ceil(self.results_count / per_page))

        self._start_page = None
        self._end_page = None
        self._iter_step = None

    def __call__(self, start=0, end=None, step=1):
        """Iterate over all pages with current paging settings.

        Parameters
        ----------
        start : int
            The first page to retrieve. Default: 0.
        end : int, optional
            The last page to retrieve. If not provided, defaults to
            ``self.page_count``.
        step : int, optional
            The step between returned pages, acts as in `range`. If not
            provided, defaults to ``1``.

        Returns
        -------
        Iterator over the requested pages.
        """
        self.page_count = int(math.ceil(self.results_count / self.per_page))
        self._start_page = start
        self._end_page = end if end else self.page_count
        self._iter_step = step

        return self.__iter__()

    def __getitem__(self, start, end=None, step=1):
        """Retrieve pages or slices by index with current paging settings.

        Parameters
        ----------
        start : int
            The first page to retrieve. Default: 0.
        end : int, optional
            The last page to retrieve. If not provided, defaults to
            ``self.page_count``.
        step : int, optional
            The step between returned pages, acts as in `range`. If not
            provided, defaults to ``1``.

        Returns
        -------
        Array of results over the requested pages.
        """
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
        """Iterate over pages of results."""
        # Recompute this in case it has changed.
        self.page_count = int(math.ceil(self.results_count / self.per_page))

        # By default, iterate over all results in the search.
        if self._start_page is None:
            self._start_page = 0

        if self._end_page is None:
            self._end_page = self.page_count

        if self._iter_step is None:
            self._iter_step = 1

        self.page_number = self._start_page
        return self

    def __next__(self):
        """Get the next page or halt iteration."""
        if self.page_number >= self._end_page:
            # Reset the defaults
            self._start_page = None
            self._end_page = None
            self._iter_step = None
            raise StopIteration

        items = get_results(self.connection, self.search_id, self)
        self.page_number += self._iter_step
        return items
