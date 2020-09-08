from .sparse_encoding import SparseMatrixSearch
from .greek_to_latin import GreekToLatinSearch

matcher_map = {}
matcher_map[SparseMatrixSearch.matcher_type] = SparseMatrixSearch
matcher_map[GreekToLatinSearch.matcher_type] = GreekToLatinSearch
