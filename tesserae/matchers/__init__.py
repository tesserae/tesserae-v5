from .default import DefaultMatcher
# from .aggregator import AggregationMatcher
from .sparse_encoding import SparseMatrixSearch

matcher_map = {}
matcher_map[SparseMatrixSearch.matcher_type] = SparseMatrixSearch
