import pytest

from test_base_tokenizer import TestBaseTokenizer

import json
import os
import re
import string
import sys

from cltk.semantics.latin.lookup import Lemmata

from tesserae.tokenizers import GreekTokenizer
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


@pytest.fixture(scope='module')
def greek_word_frequencies(greek_files):
    freqs = []
    grc = GreekTokenizer()
    for fname in greek_files:
        freq = {}
        fname = os.path.splitext(fname)[0] + '.freq_score_word'
        with open(fname, 'r') as f:
            for line in f.readlines():
                if '#' not in line:
                    word, n = re.split('[^\w' + grc.diacriticals + ']+', line,
                                       flags=re.UNICODE)[:-1]
                    freq[word] = int(n)
        freqs.append(freq)
    return freqs


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
                tokens = [t for t in tokens
                          if re.search('[' + grc.word_characters + ']+',
                                       t, flags=re.UNICODE)]

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

    def test_tokenize(self, greek_files, greek_tokens, greek_word_frequencies):
        grc = self.__test_class__()

        for k in range(len(greek_files)):
            fname = greek_files[k]
            ref_tokens = [t for t in greek_tokens[k] if 'FORM' in t]
            ref_freqs = greek_word_frequencies[k]

            t = TessFile(fname)

            token_idx = 0

            for i, line in enumerate(t.readlines(include_tag=False)):
                tokens, frequencies = grc.tokenize(line)
                tokens = [t for t in tokens
                          if re.search('[\w]',
                                       t.display, flags=re.UNICODE)]
                offset = token_idx + len(tokens)

                correct = map(lambda x: x[0].display == x[1]['DISPLAY'],
                              zip(tokens, ref_tokens[token_idx:offset]))

                if not all(correct):
                    print(fname, i, line)
                    for j in range(len(tokens)):
                        if tokens[j].display != ref_tokens[token_idx + j]['DISPLAY']:
                            print('{}->{}'.format(tokens[j].display, ref_tokens[token_idx + j]['DISPLAY']))
                            print('{}->{}'.format(tokens[j].form, ref_tokens[token_idx + j]['FORM']))

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

            grc_tokens = [t for t in grc.tokens
                          if re.search('[\w]',
                                       t.display, flags=re.UNICODE)]

            print(len(grc_tokens), len(ref_tokens))

            correct = map(lambda x: x[0].form == x[1]['FORM'],
                          zip(grc_tokens, ref_tokens))

            if not all(correct):
                for j in range(len(grc_tokens)):
                    if grc_tokens[j].form != ref_tokens[j]['FORM']:
                        print('{}->{}'.format(grc_tokens[j].form, ref_tokens[j]['FORM']))
                        print('{}->{}'.format(grc_tokens[j].display, ref_tokens[j]['DISPLAY']))

            assert all(correct)

            if '' in ref_freqs:
                del ref_freqs['']

            for key in ref_freqs:
                assert key in grc.frequencies
                assert grc.frequencies[key] == ref_freqs[key]

            diff = []
            for word in frequencies:
                if word.form not in ref_freqs and word.form != '':
                    diff.append(word.form)
            print(diff)
            assert len(diff) == 0

            assert len(frequencies) == len(ref_freqs)
            keys = sorted(list(ref_freqs.keys()))
            frequencies.sort(key=lambda x: x.form)
            correct = map(
                lambda x: x[0].form == x[1] and
                          x[0].frequency == ref_freqs[x[1]],
                zip(frequencies, keys))

            assert all(correct)

            grc.clear()
