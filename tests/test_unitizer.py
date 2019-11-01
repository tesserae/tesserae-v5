import json
import os
import re

import pytest

from tesserae.db import TessMongoConnection, Unit, Text
from tesserae.tokenizers import GreekTokenizer, LatinTokenizer
from tesserae.unitizer import Unitizer
from tesserae.utils import TessFile


@pytest.fixture(scope='module')
def unit_connection(request):
    """Create a new TessMongoConnection for this task.

    Fixtures
    --------
    request
        The configuration to connect to the MongoDB test server.
    """
    conf = request.config
    conn = TessMongoConnection(conf.getoption('db_host'),
                               conf.getoption('db_port'),
                               conf.getoption('db_user'),
                               password=conf.getoption('db_passwd',
                                                       default=None),
                               db=conf.getoption('db_name',
                                                 default=None))
    return conn


@pytest.fixture(scope='module')
def unit_tessfiles(test_data):
    """Create text entities for the test texts.

    Fixtures
    --------
    test_data
        A small set of sample texts and other entities.
    """
    tessfiles = [Text(**text) for text in test_data['texts']]
    tessfiles.sort(key=lambda x: x.path)
    return tessfiles


@pytest.fixture(scope='module')
def unitizer_inputs(unit_tessfiles, unit_connection):
    inputs = []
    for t in unit_tessfiles:
        tessfile = TessFile(t.path, metadata=t)
        if t.language == 'latin':
            tok = LatinTokenizer(unit_connection)
        if t.language == 'greek':
            tok = GreekTokenizer(unit_connection)
        tokens, tags, features = tok.tokenize(tessfile.read(), text=t)
        features.sort(key=lambda x: x.index)
        inputs.append((tokens, tags, features))
    yield inputs



@pytest.fixture(scope='module')
def correct_lines(unit_tessfiles):
    line_data = []
    for t in unit_tessfiles:
        base, _ = os.path.splitext(t.path)
        with open(base + '.line.json', 'r') as f:
            lines = json.load(f)
        with open(base + '.token.json', 'r') as f:
            tokens = json.load(f)
        with open(base + '.index_word.json', 'r') as f:
            forms = json.load(f)
        with open(base + '.index_stem.json', 'r') as f:
            stems = json.load(f)

        test_lines = []
        feature_idx = 0
        print(len(lines))
        for line in lines:
            test_line = {
                'locus': line['LOCUS'],
                'tokens': []
            }

            for t in line['TOKEN_ID']:
                if tokens[t]['TYPE'] == 'WORD':
                    test_line['tokens'].append({
                        'display': tokens[t]['DISPLAY'].strip("'"),
                        'form': forms[feature_idx][0],
                        'stem': stems[feature_idx] if not re.search(r'^[\d]+$', tokens[t]['DISPLAY']) else ['']
                    })
                    feature_idx += 1
            test_lines.append(test_line)
        line_data.append(test_lines)
    return line_data


@pytest.fixture(scope='module')
def correct_phrases(unit_tessfiles):
        phrase_data = []
        for t in unit_tessfiles:
            base, _ = os.path.splitext(t.path)
            with open(base + '.phrase.json', 'r') as f:
                phrases = json.load(f)
            with open(base + '.token.json', 'r') as f:
                tokens = json.load(f)
            with open(base + '.index_word.json', 'r') as f:
                forms = json.load(f)
            with open(base + '.index_stem.json', 'r') as f:
                stems = json.load(f)

            test_phrases = []
            feature_idx = 0
            for phrase in phrases:
                test_phrase = {
                    'locus': phrase['LOCUS'],
                    'tokens': []
                }

                for t in phrase['TOKEN_ID']:
                    if tokens[t]['TYPE'] == 'WORD':
                        if re.search(r'[\d]', tokens[t]['DISPLAY']):
                            print(tokens[t])
                        test_phrase['tokens'].append({
                            'display': tokens[t]['DISPLAY'].strip("'"),
                            'form': forms[feature_idx][0],
                            'stem': stems[feature_idx]
                        })
                        feature_idx += 1
                test_phrases.append(test_phrase)
            phrase_data.append(test_phrases)
        return phrase_data


def test_unitize(unitizer_inputs, correct_lines, correct_phrases):
    for i, indata in enumerate(unitizer_inputs):
        tokens, tags, features = indata

        feature_dict = {}
        for feature in features:
            if feature.feature in feature_dict:
                feature_dict[feature.feature].append(feature)
            else:
                feature_dict[feature.feature] = [feature]

        features = feature_dict

        unitizer = Unitizer()
        lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)

        text_correct_lines = correct_lines[i]
        assert len(lines) == len(text_correct_lines)
        for j, line in enumerate(lines):
            if isinstance(text_correct_lines[j]['locus'], str):
                assert line.tags[0] == text_correct_lines[j]['locus']
            else:
                assert line.tags == text_correct_lines[j]['locus']
            if len(line.tokens) != len(text_correct_lines[j]['tokens']):
                print(list(zip([t['display'] for t in line.tokens] + [''], [t['display'] for t in text_correct_lines[j]['tokens']])))
            assert len(line.tokens) == len(text_correct_lines[j]['tokens'])
            predicted = [t for t in line.tokens if re.search(r'[\w]', t['display'])]
            for k in range(len(predicted)):
                token = predicted[k]
                correct = text_correct_lines[j]['tokens'][k]

                assert token['display'] == correct['display']

                if token['features']['form'][0] > -1:
                    form = feature_dict['form'][token['features']['form'][0]].token
                    lemmata = [feature_dict['lemmata'][l].token for l in token['features']['lemmata']]
                else:
                    form = ''
                    lemmata = ['']
                if form != correct['form']:
                    print(token, correct)
                    print(form, correct['form'])
                assert form == correct['form']
                assert len(lemmata) == len(correct['stem'])
                print(token['display'], form, lemmata, correct['display'], correct['form'], correct['stem'])
                assert all(map(lambda x: x in correct['stem'], lemmata))

    text_correct_phrases = correct_phrases[i]
    assert len(phrases) == len(text_correct_phrases)
    for j, phrase in enumerate(phrases[:-1]):
        if isinstance(text_correct_phrases[j]['locus'], str):
            assert phrase.tags[0] == text_correct_phrases[j]['locus']
        else:
            assert phrase.tags == text_correct_phrases[j]['locus']
        if len(phrase.tokens) != len(text_correct_phrases[j]['tokens']):
            print(list(zip([t['display'] for t in phrase.tokens] + [''], [t['display'] for t in text_correct_phrases[j]['tokens']])))
        assert len(phrase.tokens) == len(text_correct_phrases[j]['tokens'])
        predicted = [t for t in phrase.tokens if re.search(r'[\w]', t['display'])]
        for k in range(len(predicted)):
            token = predicted[k]
            correct = text_correct_phrases[j]['tokens'][k]

            assert token['display'] == correct['display']

            if token['features']['form'][0] > -1:
                form = feature_dict['form'][token['features']['form'][0]].token
                lemmata = [feature_dict['lemmata'][l].token for l in token['features']['lemmata']]
            else:
                form = ''
                lemmata = ['']
            if form != correct['form']:
                print(token, correct)
                print(form, correct['form'])
            assert form == correct['form']
            assert len(lemmata) == len(correct['stem'])
            print(token['display'], form, lemmata, correct['display'], correct['form'], correct['stem'])
            assert all(map(lambda x: x in correct['stem'], lemmata))
