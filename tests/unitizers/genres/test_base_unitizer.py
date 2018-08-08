import pytest

from tesserae.unitizers.genres import BaseUnitizer
from tesserae.tokenizers.languages import GreekTokenizer, LatinTokenizer


class TestBaseUnitizer(object):
    __test_unitizer__ = BaseUnitizer

    def get_tokenizer(self, fname):
        if 'grc' in fname:
            return GreekTokenizer()
        else:
            return LatinTokenizer()

    def test_unitize_lines(self):
        if self.__test_unitizer__ is BaseUnitizer:
            with pytest.raises(NotImplementedError):
                u = self.__test_unitizer__()
                u.unitize_lines(None, None)

    def test_unitize_phrases(self):
        if self.__test_unitizer__ is BaseUnitizer:
            with pytest.raises(NotImplementedError):
                u = self.__test_unitizer__()
                u.unitize_phrases(None, None)
