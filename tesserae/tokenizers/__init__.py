from .base import BaseTokenizer
from .english import EnglishTokenizer
from .greek import GreekTokenizer
from .latin import LatinTokenizer

tokenizer_map = {}
tokenizer_map['greek'] = GreekTokenizer
tokenizer_map['latin'] = LatinTokenizer
tokenizer_map['english'] = EnglishTokenizer

__all__ = [
    'BaseTokenizer', 'GreekTokenizer', 'LatinTokenizer', 'EnglishTokenizer'
]
