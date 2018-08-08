import pytest

import re

from test_base_unitizer import TestBaseUnitizer

from tesserae.unitizers.genres import PoetryUnitizer
from tesserae.utils import TessFile
from tesserae.tokenizers.languages import GreekTokenizer, LatinTokenizer


class TestPoetryUnitizer(TestBaseUnitizer):
    __test_unitizer__ = PoetryUnitizer

    def test_unitize_lines(self, poetry_files, poetry_lines):
        for i, f in enumerate(poetry_files):
            tokenizer = self.get_tokenizer(f)
            t = TessFile(f)
            u = self.__test_unitizer__()

            lines = u.unitize_lines(t, tokenizer)
            for j, l in enumerate(lines):
                #print(j)
                assert l.raw == re.sub(r'[\u201c\u201d]', '"',
                                       poetry_lines[i][j]['raw'].strip(),
                                       flags=re.UNICODE)
                for k, form in enumerate(l.tokens):
                    assert form == poetry_lines[i][j]['forms'][k]

    def test_unitize_phrases(self, poetry_files, poetry_phrases):
        for i, f in enumerate(poetry_files):
            tokenizer = self.get_tokenizer(f)
            t = TessFile(f)
            u = self.__test_unitizer__()

            phrases = u.unitize_lines(t, tokenizer)
            for j, l in enumerate(phrases):
                assert l.raw == poetry_phrases[i][j]['raw'].strip()
                for k, form in l.tokens:
                    assert form == poetry_phrases[i][j]['forms'][k]
