import pytest

from test_base_tokenizer import TestBaseTokenizer

import json
import os
import re
import string
import sys

from cltk.semantics.latin.lookup import Lemmata
from cltk.stem.latin.j_v import JVReplacer

from tesserae.tokenizers.languages import LatinTokenizer
from tesserae.utils import TessFile


@pytest.fixture(scope='session')
def latin_tokens(latin_files):
    tokens = []
    for fname in latin_files:
        fname = os.path.splitext(fname)[0] + '.tokens.json'
        with open(fname, 'r') as f:
            ts = [t for t in json.load(f)]
            tokens.append(ts)
    return tokens


class TestLatinTokenizer(TestBaseTokenizer):
    __test_class__ = LatinTokenizer

    def test_init(self):
        t = self.__test_class__()
        assert hasattr(t, 'jv_replacer')
        assert isinstance(t.jv_replacer, JVReplacer)
        assert hasattr(t, 'lemmatizer')
        assert isinstance(t.lemmatizer, Lemmata)

    def test_normalize(self, latin_files, latin_tokens):
        la = self.__test_class__()

        for i in range(len(latin_files)):
            fname = latin_files[i]
            ref_tokens = [t for t in latin_tokens[i] if 'FORM' in t]

            t = TessFile(fname)

            token_idx = 0

            for i, line in enumerate(t.readlines(include_tag=False)):
                tokens = [t for t in la.normalize(line)
                    if re.search(r'[a-zA-Z]+', t, flags=re.UNICODE) is not None]

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

            token_idx = 0
            for i, token in enumerate(t.read_tokens(include_tag=False)):
                tokens = [t for t in la.normalize(token)
                    if re.search(r'[a-zA-Z]+', t, flags=re.UNICODE) is not None]

                offset = token_idx + len(tokens)

                correct = map(lambda x: x[0] == x[1]['FORM'],
                              zip(tokens, ref_tokens[token_idx:offset]))
                assert all(correct)

                token_idx = offset

    def test_tokenize(self, latin_files, latin_tokens):
        la = self.__test_class__()

        for k in range(len(latin_files)):
            fname = latin_files[k]
            ref_tokens = [t for t in latin_tokens[k] if 'FORM' in t]

            t = TessFile(fname)

            token_idx = 0

            for i, line in enumerate(t.readlines(include_tag=False)):
                tokens, frequencies = la.tokenize(line)
                tokens = [t for t in tokens
                          if re.search(r'^[a-zA-Z]+$', t.display,
                                       flags=re.UNICODE)]

                offset = token_idx + len(tokens)

                correct = map(lambda x: x[0].display == x[1]['DISPLAY'],
                              zip(tokens, ref_tokens[token_idx:offset]))

                if not all(correct):
                    print(fname, i, line)
                    for j in range(len(tokens)):
                        if tokens[j].display != ref_tokens[token_idx + j]['DISPLAY']:
                            print('{}->{}'.format(tokens[j].display, ref_tokens[token_idx + j]['DISPLAY']))

                assert all(correct)

                correct = map(lambda x: x[0].form == x[1]['FORM'],
                              zip(tokens, ref_tokens[token_idx:offset]))

                if not all(correct):
                    print(fname, i, line)
                    for j in range(len(tokens)):
                        if tokens[j].form != ref_tokens[token_idx + j]['FORM']:
                            print('{}->{}'.format(tokens[j].form, ref_tokens[token_idx + j]['FORM']))

                assert all(correct)

                token_idx = offset

            la_tokens = [t for t in la.tokens
                      if re.search(r'^[a-zA-Z]+$', t.display, flags=re.UNICODE)]

            correct = map(lambda x: x[0].form == x[1]['FORM'],
                          zip(la_tokens, ref_tokens))

            print(len(la_tokens), len(ref_tokens))

            if not all(correct):
                for j in range(len(la_tokens)):
                    if tokens[j].form != ref_tokens[j]['FORM']:
                        print('{}->{}'.format(la_tokens[j].form, ref_tokens[token_idx + j]['FORM']))

            assert all(correct)

            la.clear()
