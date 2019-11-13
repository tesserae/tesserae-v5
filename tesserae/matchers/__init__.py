from .default import DefaultMatcher
# from .aggregator import AggregationMatcher
from .sparse_encoding import SparseMatrixSearch

search_types = {
    'original': SparseMatrixSearch
}
