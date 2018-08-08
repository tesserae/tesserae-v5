import pytest

import random

from tesserae.tokenizers.languages import BaseTokenizer


class TestBaseTokenizer(object):
    __test_class__ = BaseTokenizer

    def test_normalize(self):
        test_words = ['foo', 'Bar', 'BAZ', 'FoO bAr BaZ']
        punct = '.,?:;\'"!'
        whitespace = ' \t\n\r'

        t = self.__test_class__()

        # Ensure that an empty string is returned as an empty list
        assert t.normalize('') == []

        # Ensure that an empty list is returned as an empty list
        assert t.normalize([]) == []

        # Ensure that standard words are converted to lowercase and split if
        # spaces exist in the string.
        for word in test_words[:-1]:
            assert t.normalize(word) == [word.lower()]

        assert t.normalize(test_words[-1]) == [w.lower()
                                               for w in test_words[-1].split()]

        # Ensure that normalization is applied to every string in a list
        for i in range(len(test_words) - 1):
            assert t.normalize(test_words[:i + 1]) == \
                   [w.lower()for w in test_words[:i + 1]]

        # Ensure that strings containing only whitespace, punctuation, or
        # digits are returned as empty lists
        for i in range(len(punct)):
            assert t.normalize(punct[i]) == []
            assert t.normalize(punct[:i + 1]) == []

        for i in range(len(whitespace)):
            assert t.normalize(whitespace[i]) == []
            assert t.normalize(whitespace[:i + 1]) == []

        # Ensure that tokens with punctuation and whitespace are returned
        # lowercase with punct and whitespace removed
        for word in test_words[:-1]:
            for _ in range(100):
                p = punct[random.randint(0, len(punct) - 1)]
                w = whitespace[random.randint(0, len(whitespace) - 1)]

                assert t.normalize(w + word) == [word.lower()]
                assert t.normalize(p + word) == [word.lower()]
                assert t.normalize(word + w) == [word.lower()]
                assert t.normalize(word + p) == [word.lower()]
                assert t.normalize(p + word + w) == [word.lower()]
                assert t.normalize(w + word + p) == [word.lower()]
                assert t.normalize(p + w + word) == [word.lower()]
                assert t.normalize(w + p + word) == [word.lower()]
                assert t.normalize(word + p + w) == [word.lower()]
                assert t.normalize(word + w + p) == [word.lower()]
                assert t.normalize(p + w + word + p + w) == [word.lower()]
                assert t.normalize(w + p + word + w + p) == [word.lower()]
                assert t.normalize(w + p + word + p + w) == [word.lower()]
                assert t.normalize(p + w + word + w + p) == [word.lower()]




    def test_featurize(self):
        t = self.__test_class__()

        if self.__test_class__ is BaseTokenizer:
            with pytest.raises(NotImplementedError):
                t.featurize('foo')
