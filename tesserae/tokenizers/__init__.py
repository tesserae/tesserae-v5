from .base import BaseTokenizer
from .greek import GreekTokenizer
from .latin import LatinTokenizer

tokenizer_map = {}
tokenizer_map['greek'] = GreekTokenizer
tokenizer_map['latin'] = LatinTokenizer

__all__ = ['BaseTokenizer', 'GreekTokenizer', 'LatinTokenizer']
