import pytest

import json
import os
import re
import string
import sys

from cltk.semantics.latin.lookup import Lemmata

from tesserae.db.entities.text import Text
from tesserae.tokenizers import GreekTokenizer
from tesserae.utils import TessFile


@pytest.fixture(scope='module')
def greek_tessfiles(test_data, token_connection):
    # Get the test data and filter for Greek texts only.
    tessfiles = [t for t in test_data['texts'] if t['language'] == 'greek']
    tessfiles = [Text(**text) for text in tessfiles]

    # Prep the database with the text metadata
    token_connection.insert(tessfiles)

    # Create file readers for each text
    tessfiles = [TessFile(text.path, metadata=text) for text in tessfiles]

    yield sorted(tessfiles, key=lambda x: x.metadata.path)

    token_connection.delete([t.metadata for t in tessfiles])


@pytest.fixture(scope='module')
def greek_tokens(greek_files):
    greek_tokens = []

    tessfiles = sorted(
                    filter(lambda x: os.path.splitext(x)[1] == '.tess',
                           greek_files))

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
            if tokens[i]['TYPE'] == 'WORD':
                text_tokens.append({
                    'display': tokens[i]['DISPLAY'],
                    'form': forms[idx][0],
                    'lemmata': stems[idx] if forms[idx][0] else ['']
                    })
                idx += 1
        greek_tokens.append(text_tokens)
    return greek_tokens


@pytest.fixture(scope='module')
def greek_word_frequencies(greek_files):
    freqs = []
    for fname in greek_files:
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
    t = GreekTokenizer(token_connection)
    assert t.connection is token_connection
    assert hasattr(t, 'lemmatizer')
    assert isinstance(t.lemmatizer, Lemmata)


def test_normalize(token_connection, greek_tessfiles, greek_tokens):
    grc = GreekTokenizer(token_connection)

    for i, tessfile in enumerate(greek_tessfiles):
        correct_tokens = [t for t in greek_tokens[i] if t['form']]
        tokens, tags = grc.normalize(tessfile.read())
        tokens = [t for t in tokens if re.search(r'[\w]+', t)]
        correct = map(lambda x: x[0] == x[1]['form'],
                      zip(tokens, correct_tokens))
        for j, c in enumerate(correct):
            if not c:
                print(j, tokens[j], correct_tokens[j])
                break
        assert all(correct)

        for i, line in enumerate(tessfile.readlines()):
            correct_tag = line[:line.find('>') + 1]
            assert tags[i] == correct_tag


def test_tokenize(token_connection, greek_tessfiles, greek_tokens):
    grc = GreekTokenizer(token_connection)

    for i, tessfile in enumerate(greek_tessfiles):
        print(tessfile.metadata.title)
        tokens, tags, features = grc.tokenize(
            tessfile.read(), text=tessfile.metadata)
        tokens = [t for t in tokens if re.search(r'[\w]+', t.display)]

        for j, token in enumerate(tokens):
            # Detect all connected
            assert token.display == greek_tokens[i][j]['display']
            # if tessfile.metadata.title == 'gorgias':
            #     print(token.display, greek_tokens[i][j])
            # print(token.display, token.features['form'].token, [t.token for t in token.features['lemmata']])
            # print(greek_tokens[i][j])
            assert token.features['form'].token == greek_tokens[i][j]['form']
            assert all([
                any(
                    map(lambda x: lemma.token == x,
                        greek_tokens[i][j]['lemmata']))
                for lemma in token.features['lemmata']])


def test_featurize(token_connection):
    pass
