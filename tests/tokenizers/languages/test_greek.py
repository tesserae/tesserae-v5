import pytest

from test_base_tokenizer import TestBaseTokenizer

import json
import os
import re
import string
import sys

from cltk.semantics.latin.lookup import Lemmata

from tesserae.tokenizers.languages import GreekTokenizer
from tesserae.utils import TessFile


@pytest.fixture(scope='session')
def greek_tokens(greek_files):
    tokens = []
    for f in greek_files:
        root, ext = os.path.splitext(f)
        fname = root + '.tokens' + '.json'
        with open(fname, 'r') as f:
            ts = [t for t in json.load(f) if t['TYPE'] == 'WORD']
            tokens.append(ts)
    return tokens


class TestGreekTokenizer(TestBaseTokenizer):
    __test_class__ = GreekTokenizer

    def test_init(self):
        t = self.__test_class__()
        assert hasattr(t, 'diacriticals')
        assert isinstance(t.diacriticals, str)
        assert hasattr(t, 'vowels')
        assert isinstance(t.vowels, str)
        assert hasattr(t, 'grave')
        assert isinstance(t.grave, str)
        assert hasattr(t, 'acute')
        assert isinstance(t.acute, str)
        assert hasattr(t, 'diacrit_sub1')
        assert isinstance(t.diacrit_sub1, str)
        assert hasattr(t, 'diacrit_sub2')
        assert isinstance(t.diacrit_sub2, str)
        assert hasattr(t, 'lemmatizer')
        assert isinstance(t.lemmatizer, Lemmata)

    def test_normalize(self, greek_files, greek_tokens):
        grc = self.__test_class__()

        for i in range(len(greek_files)):
            fname = greek_files[i]
            ref_tokens = [t for t in greek_tokens[i] if t['FORM'] != '']

            t = TessFile(fname)

            token_idx = 0

            for i, line in enumerate(t.readlines(include_tag=False)):
                tokens = [t for t in grc.normalize(line)]

                offset = token_idx + len(tokens)

                correct = map(lambda x: x[0] == x[1]['FORM'],
                              zip(tokens, ref_tokens[token_idx:offset]))

                if not all(correct):
                    print(fname, i, line)
                    print(ref_tokens[token_idx:offset])
                    for j in range(len(tokens)):
                        if tokens[j] != ref_tokens[token_idx + j]['FORM']:
                            print('{}->{}'.format(tokens[j], ref_tokens[token_idx + j]['FORM']))

                assert all(correct)

                token_idx = offset

            # token_idx = 0
            # for i, token in enumerate(t.read_tokens(include_tag=False)):
            #     tokens = [t for t in grc.normalize(token)]
            #
            #     offset = token_idx + len(tokens)
            #
            #     correct = map(lambda x: x[0] == x[1]['FORM'],
            #                   zip(tokens, ref_tokens[token_idx:offset]))
            #     assert all(correct)
            #
            #     token_idx = offset
