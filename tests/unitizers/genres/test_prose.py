import pytest

import re

from test_base_unitizer import TestBaseUnitizer

from tesserae.unitizers.genres import ProseUnitizer
from tesserae.utils import TessFile
from tesserae.tokenizers.languages import GreekTokenizer, LatinTokenizer


class TestProseUnitizer(TestBaseUnitizer):
    __test_unitizer__ = ProseUnitizer

    def test_unitize_lines(self, prose_files, prose_lines):
        for i, f in enumerate(prose_files):
            tokenizer = self.get_tokenizer(f)
            t = TessFile(f)
            u = self.__test_unitizer__()

            lines = u.unitize_lines(t, tokenizer)
            for j, l in enumerate(lines):
                #print(j)
                assert l.raw == re.sub(r'[\u201c\u201d]', '"',
                                       prose_lines[i][j]['raw'].strip(),
                                       flags=re.UNICODE)
                for k, form in enumerate(l.tokens):
                    assert form == prose_lines[i][j]['forms'][k]

    def test_unitize_phrases(self, prose_files, prose_phrases):
        for i, f in enumerate(prose_files):
            tokenizer = self.get_tokenizer(f)
            t = TessFile(f)
            u = self.__test_unitizer__()

            phrases = u.unitize_lines(t, tokenizer)
            for j, l in enumerate(phrases):
                assert l.raw == prose_phrases[i][j]['raw'].strip()
                for k, form in l.tokens:
                    assert form == prose_phrases[i][j]['forms'][k]
