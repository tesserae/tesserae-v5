import pytest

import json
import os
import re
import string
import sys

from cltk.semantics.latin.lookup import Lemmata
from cltk.stem.latin.j_v import JVReplacer

from tesserae.db.entities.text import Text
from tesserae.tokenizers import LatinTokenizer
from tesserae.utils import TessFile


@pytest.fixture(scope='module')
def latin_tessfiles(test_data, token_connection):
    # Get the test data and filter for Latin texts only.
    tessfiles = [t for t in test_data['texts'] if t['language'] == 'latin']
    tessfiles = [Text(**text) for text in tessfiles]

    # Prep the database with the text metadata
    token_connection.insert(tessfiles)

    # Create file readers for each text
    tessfiles = [TessFile(text.path, metadata=text) for text in tessfiles]

    yield sorted(tessfiles, key=lambda x: x.metadata.path)

    token_connection.delete([t.metadata for t in tessfiles])


@pytest.fixture(scope='module')
def latin_tokens(latin_files):
    latin_tokens = []

    tessfiles = sorted(
                    filter(lambda x: os.path.splitext(x)[1] == '.tess',
                           latin_files))

    for fname in tessfiles:
        text = os.path.splitext(fname)[0]

        with open(text + '.token.json', 'r') as f:
            tokens = json.load(f)

        with open(text + '.index_word.json', 'r') as f:
            forms = json.load(f)

        with open(text + '.index_stem.json', 'r') as f:
            stems = json.load(f)

        with open(text + '.index_3gr.json', 'r') as f:
            trigrams = json.load(f)

        idx = 0
        text_tokens = []
        for i in range(len(tokens)):
            if tokens[i]['TYPE'] == 'WORD' and not re.search(r'[\d]', tokens[i]['DISPLAY']):
                text_tokens.append({
                    'display': tokens[i]['DISPLAY'],
                    'form': forms[idx][0],
                    'lemmata': stems[idx]
                    })
                idx += 1
        latin_tokens.append(text_tokens)
    return latin_tokens


@pytest.fixture(scope='module')
def latin_word_frequencies(latin_files):
    freqs = []
    for fname in latin_files:
        freq = {}
        fname = os.path.splitext(fname)[0] + '.freq_score_word'
        with open(fname, 'r') as f:
            for line in f.readlines():
                if '#' not in line:
                    word, n = line.strip().split()
                    freq[word] = int(n)
        freqs.append(freq)
    return freqs


def test_init(token_connection):
    t = LatinTokenizer(token_connection)
    assert t.connection is token_connection
    assert hasattr(t, 'jv_replacer')
    assert isinstance(t.jv_replacer, JVReplacer)
    assert hasattr(t, 'lemmatizer')
    assert isinstance(t.lemmatizer, Lemmata)


def test_normalize(token_connection, latin_tessfiles, latin_tokens):
    la = LatinTokenizer(token_connection)

    for i, tessfile in enumerate(latin_tessfiles):
        tokens, tags = la.normalize(tessfile.read())
        correct = map(lambda x: x[0] == x[1]['form'],
                      zip(tokens, latin_tokens[i]))
        for j, c in enumerate(correct):
            if not c:
                print(latin_tokens[i][j])
                break
        assert all(correct)

        for i, line in enumerate(tessfile.readlines()):
            correct_tag = line[:line.find('>') + 1]
            assert tags[i] == correct_tag


def test_tokenize(token_connection, latin_tessfiles, latin_tokens):
    la = LatinTokenizer(token_connection)

    for i, tessfile in enumerate(latin_tessfiles):
        tokens, tags, features = la.tokenize(
            tessfile.read(), text=tessfile.metadata)

        tokens = filter(lambda x: re.search(r'[\w]', x.display), tokens)

        for j, token in enumerate(tokens):
            # Detect all connected
            assert token.display == latin_tokens[i][j]['display']
            assert token.features['form'].token == latin_tokens[i][j]['form']
            assert all([
                any(
                    map(lambda x: lemma.token == x,
                        latin_tokens[i][j]['lemmata']))
                for lemma in token.features['lemmata']])

        for i, line in enumerate(tessfile.readlines()):
            correct_tag = line[:line.find('>') + 1].split()[-1].strip('>')
            assert tags[i] == correct_tag


def test_featurize(token_connection):
    pass
