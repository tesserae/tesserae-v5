from collections import defaultdict
import json
import os
import re

import pytest

from tesserae.db import TessMongoConnection, Text
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
def unit_tessfiles(mini_greek_metadata, mini_latin_metadata):
    """Create text entities for the test texts.

    Fixtures
    --------
    test_data
        A small set of sample texts and other entities.
    """
    tessfiles = []
    for metadata in mini_greek_metadata:
        tessfiles.append(Text.json_decode(metadata))
    for metadata in mini_latin_metadata:
        tessfiles.append(Text.json_decode(metadata))
    tessfiles.sort(key=lambda x: x.path)
    return tessfiles


@pytest.fixture(scope='module')
def unitizer_inputs(unit_tessfiles, unit_connection):
    inputs = []
    tokenizer_selector = {
        'latin': LatinTokenizer(unit_connection),
        'greek': GreekTokenizer(unit_connection)
    }
    for t in unit_tessfiles:
        tessfile = TessFile(t.path, metadata=t)
        tokens, tags, features = tokenizer_selector[t.language].tokenize(
                tessfile.read(), text=t)
        features.sort(key=lambda x: x.index)
        inputs.append((tokens, tags, features))
    yield inputs


def _extract_unit_information(units, tokens, reverse_stems):
    test_units = []
    for unit in units:
        test_unit = {
            'locus': unit['LOCUS'],
            'tokens': []
        }

        for t in unit['TOKEN_ID']:
            cur_token = tokens[t]
            if cur_token['TYPE'] == 'WORD':
                cur_token_display = cur_token['DISPLAY']
                cur_token_form = cur_token['FORM']
                if re.search(r'[\d]', cur_token_display):
                    print(tokens[t])
                test_unit['tokens'].append({
                    # ignore elision mark
                    'display': cur_token_display.strip("'"),
                    'form': cur_token_form,
                    'stem': reverse_stems[t]
                })
        test_units.append(test_unit)
    return test_units


@pytest.fixture(scope='module')
def correct_units(unit_tessfiles):
    results = {'lines': [], 'phrases': []}
    for t in unit_tessfiles:
        base, _ = os.path.splitext(t.path)
        with open(base + '.line.json', 'r') as f:
            lines = json.load(f)
        with open(base + '.phrase.json', 'r') as f:
            phrases = json.load(f)
        with open(base + '.token.json', 'r') as f:
            tokens = json.load(f)
        with open(base + '.index_stem.json', 'r') as f:
            stems = json.load(f)
        reverse_stems = defaultdict(list)
        for stem, token_ids in stems.items():
            for token_id in token_ids:
                reverse_stems[int(token_id)].append(stem)

        results['lines'].append(_extract_unit_information(
            lines, tokens, reverse_stems))
        results['phrases'].append(_extract_unit_information(
            phrases, tokens, reverse_stems))
    return results


WORD_PATTERN = re.compile(r'[\w]', flags=re.UNICODE)


def test_unitize(unitizer_inputs, correct_units):
    correct_lines = correct_units['lines']
    correct_phrases = correct_units['phrases']
    for i, indata in enumerate(unitizer_inputs):
        tokens, tags, features = indata

        feature_dict = {}
        for feature in features:
            if feature.feature in feature_dict:
                feature_dict[feature.feature][feature.index] = feature
            else:
                feature_dict[feature.feature] = {feature.index: feature}

        features = feature_dict

        unitizer = Unitizer()
        lines, phrases = unitizer.unitize(
            tokens, tags, tokens[0].text)

        text_correct_lines = correct_lines[i]
        assert len(lines) == len(text_correct_lines)
        for j, line in enumerate(lines):
            line_snippet = line.snippet
            assert WORD_PATTERN.search(line_snippet[0]) is not None
            assert not line_snippet.endswith(' / ')
            if isinstance(text_correct_lines[j]['locus'], str):
                assert line.tags[0] == text_correct_lines[j]['locus']
            else:
                assert line.tags == text_correct_lines[j]['locus']
            if len(line.tokens) != len(text_correct_lines[j]['tokens']):
                print(list(zip(
                    [t['display'] for t in line.tokens] + [''],
                    [t['display'] for t in text_correct_lines[j]['tokens']])))
            assert len(line.tokens) == len(text_correct_lines[j]['tokens'])
            predicted = [
                t for t in line.tokens if re.search(r'[\w]', t['display'])]
            for k in range(len(predicted)):
                token = predicted[k]
                correct = text_correct_lines[j]['tokens'][k]

                assert token['display'] == correct['display']

                if token['features']['form'][0] > -1:
                    cur_form_index = token['features']['form'][0]
                    form = feature_dict['form'][cur_form_index].token
                    lemmata = [
                        feature_dict['lemmata'][l].token
                        for l in token['features']['lemmata']]
                else:
                    form = ''
                    lemmata = ['']
                if form != correct['form']:
                    print(token, correct)
                    print(form, correct['form'])
                assert form == correct['form']
                assert len(lemmata) == len(correct['stem'])
                assert all(map(lambda x: x in correct['stem'], lemmata))

        text_correct_phrases = correct_phrases[i]
        assert len(phrases) == len(text_correct_phrases)
        for j, phrase in enumerate(phrases):
            assert WORD_PATTERN.search(phrase.snippet[0]) is not None
            if isinstance(text_correct_phrases[j]['locus'], str):
                assert phrase.tags[0] == text_correct_phrases[j]['locus']
            else:
                assert phrase.tags == text_correct_phrases[j]['locus']
            if len(phrase.tokens) != len(text_correct_phrases[j]['tokens']):
                print(list(zip(
                    [t['display'] for t in phrase.tokens] + [''],
                    [t['display']
                     for t in text_correct_phrases[j]['tokens']])))
            assert len(phrase.tokens) == len(
                text_correct_phrases[j]['tokens'])
            predicted = [
                t for t in phrase.tokens if re.search(r'[\w]', t['display'])]
            for k in range(len(predicted)):
                token = predicted[k]
                correct = text_correct_phrases[j]['tokens'][k]

                assert token['display'] == correct['display']

                if token['features']['form'][0] > -1:
                    cur_form_index = token['features']['form'][0]
                    form = feature_dict['form'][cur_form_index].token
                    lemmata = [
                        feature_dict['lemmata'][l].token
                        for l in token['features']['lemmata']]
                else:
                    form = ''
                    lemmata = ['']
                if form != correct['form']:
                    print(token, correct)
                    print(form, correct['form'])
                assert form == correct['form']
                assert len(lemmata) == len(correct['stem'])
                assert all(map(lambda x: x in correct['stem'], lemmata))


def test_unitize_linebreak_file(unit_connection, tessfiles_latin_path):
    tokenizer = LatinTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_latin_path.joinpath('test.linebreak.tess')),
        language='latin')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    assert len(lines) == 1
    first_tag = phrases[0].tags[0]
    for phrase in phrases[1:]:
        assert phrase.tags[0] == first_tag


def test_unitize_doublelinebreak_file(unit_connection, tessfiles_latin_path):
    tokenizer = LatinTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_latin_path.joinpath('test.doublelinebreak.tess')),
        language='latin')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    assert len(lines) == 1
    first_tag = phrases[0].tags[0]
    for phrase in phrases[1:]:
        assert phrase.tags[0] == first_tag


def test_unitize_nonumber_file(unit_connection, tessfiles_latin_path):
    tokenizer = LatinTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_latin_path.joinpath('test.nonumber.tess')),
        language='latin')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    assert len(lines) == 1


def test_unitize_nopunctuation_file(unit_connection, tessfiles_latin_path):
    # when there is no ending punctuation despite coming to the end of a poem
    # and another poem starts after a blank line
    tokenizer = LatinTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_latin_path.joinpath('test.nopunctuation.tess')),
        language='latin')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    assert len(lines) == 68
    for prev_phrase, cur_phrase in zip(phrases[:-1], phrases[1:]):
        if '2.13' in prev_phrase.tags[0] and '2.14' in cur_phrase.tags[0]:
            assert prev_phrase.snippet == (
                'quin et Prometheus et Pelopis parens / '
                'dulci laborem decipitur sono / '
                'nec curat Orion leones / '
                'aut timidos agitare lyncas / '
                'Eheu fugaces, Postume, Postume, / '
                'labuntur anni nec pietas moram / '
                'rugis et instanti senectae / '
                'adferet indomitaeque morti, / '
                'non, si trecenis quotquot eunt dies, / '
                'amice, places inlacrimabilem / '
                'Plutona tauris, qui ter amplum / '
                'Geryonen Tityonque tristi / '
                'conpescit unda, scilicet omnibus / '
                'quicumque terrae munere vescimur / '
                'enaviganda, sive reges / '
                'sive inopes erimus coloni. / '
            )
            assert cur_phrase.snippet == (
                'frustra cruento Marte carebimus / '
                'fractisque rauci fluctibus Hadriae, / '
                'frustra per autumnos nocentem / '
                'corporibus metuemus Austrum: / '
            )
            break


def test_unitize_notag_file(unit_connection, tessfiles_latin_path):
    tokenizer = LatinTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_latin_path.joinpath('test.notag.tess')),
        language='latin')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    assert len(lines) == 1


def test_unitize_linebreak_end(unit_connection, tessfiles_latin_path):
    tokenizer = LatinTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_latin_path.joinpath('test.linebreak_end.tess')),
        language='latin')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    print('# lines')
    for line in lines:
        print(line.snippet)
    print('# phrases')
    for phrase in phrases:
        print(phrase.snippet)
    assert len(lines) == 2


def test_unitize_greek_in_latin(unit_connection, tessfiles_latin_path):
    tokenizer = LatinTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_latin_path.joinpath('test.greek_in_latin.tess')),
        language='latin')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    forms = {f.index: f.token
             for f in features if f.feature == 'form'}
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    for phrase in phrases:
        for t in phrase.tokens:
            cur_form = t['features']['form'][0]
            if cur_form != -1:
                normalized = tokenizer.normalize(t['display'])[0][0]
                assert normalized == forms[cur_form], phrase.snippet


def test_unitize_numbers_in_latin(unit_connection, tessfiles_latin_path):
    tokenizer = LatinTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_latin_path.joinpath('test.numbers_in_latin.tess')),
        language='latin')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    forms = {f.index: f.token
             for f in features if f.feature == 'form'}
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    for phrase in phrases:
        for t in phrase.tokens:
            cur_form = t['features']['form'][0]
            if cur_form != -1:
                normalized = tokenizer.normalize(t['display'])[0][0]
                assert normalized == forms[cur_form], phrase.snippet


def test_unitize_diacrit_in_latin(unit_connection, tessfiles_latin_path):
    tokenizer = LatinTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_latin_path.joinpath('test.diacrit_in_latin.tess')),
        language='latin')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    forms = {f.index: f.token
             for f in features if f.feature == 'form'}
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    for phrase in phrases:
        for t in phrase.tokens:
            cur_form = t['features']['form'][0]
            if cur_form != -1:
                normalized = tokenizer.normalize(t['display'])[0][0]
                assert normalized == forms[cur_form], phrase.snippet


def test_unitize_lone_diacrit_file(unit_connection, tessfiles_greek_path):
    tokenizer = GreekTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_greek_path.joinpath('test.lone_diacrit.tess')),
        language='greek')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    assert len(lines) == 1


def test_unitize_elision_file(unit_connection, tessfiles_greek_path):
    tokenizer = GreekTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_greek_path.joinpath('test.elision.tess')),
        language='greek')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    assert len(lines) == 1


def test_unitize_middot_file(unit_connection, tessfiles_greek_path):
    tokenizer = GreekTokenizer(unit_connection)
    t = Text(
        path=str(tessfiles_greek_path.joinpath('test.middot.tess')),
        language='greek')
    tessfile = TessFile(t.path, metadata=t)
    unitizer = Unitizer()
    tokens, tags, features = tokenizer.tokenize(
            tessfile.read(), text=t)
    lines, phrases = unitizer.unitize(tokens, tags, tokens[0].text)
    assert len(lines) == 1
    assert len(lines[0].tokens) == 7
