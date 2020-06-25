import itertools
import uuid

import math
import pytest

from tesserae.db import Feature, Search, Text, \
                        TessMongoConnection
from tesserae.matchers.sparse_encoding import \
        SparseMatrixSearch, get_inverse_text_frequencies, \
        get_corpus_frequencies
from tesserae.matchers.text_options import TextOptions
from tesserae.tokenizers import LatinTokenizer
from tesserae.unitizer import Unitizer
from tesserae.utils import ingest_text
from tesserae.utils.delete import obliterate
from tesserae.utils.tessfile import TessFile


@pytest.fixture(scope='session')
def punctpop(request, mini_punctuation_metadata):
    conn = TessMongoConnection('localhost', 27017, None, None, 'puncttess')
    for metadata in mini_punctuation_metadata:
        text = Text.json_decode(metadata)
        ingest_text(conn, text)
    yield conn
    obliterate(conn)


def test_mini_text_frequencies(
        minipop, mini_latin_metadata,
        mini_greek_metadata, v3checker):
    all_text_metadata = [m for m in itertools.chain.from_iterable(
        [mini_latin_metadata, mini_greek_metadata])]
    title2id = {t.title: t.id for t in minipop.find(
        Text.collection, title=[m['title'] for m in all_text_metadata])}
    for metadata in all_text_metadata:
        v3freqs = v3checker.load_v3_mini_text_stem_freqs(minipop, metadata)
        text_id = title2id[metadata['title']]
        v5freqs = get_inverse_text_frequencies(minipop, 'lemmata', text_id)
        for form_index, freq in v5freqs.items():
            assert form_index in v3freqs
            assert math.isclose(v3freqs[form_index], 1.0 / freq)


def test_mini_latin_search_text_freqs(minipop, mini_latin_metadata, v3checker):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'line'),
        TextOptions(texts[1], 'line'),
        'lemmata',
        stopwords=['et', 'neque', 'qui'],
        stopword_basis='texts', score_basis='stem',
        freq_basis='texts', max_distance=10,
        distance_basis='frequency', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, results_id, texts[0].path,
                                   'mini_latin_results.tab')


def test_mini_greek_search_text_freqs(minipop, mini_greek_metadata, v3checker):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_greek_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'phrase'),
        TextOptions(texts[1], 'phrase'),
        'lemmata',
        stopwords=[
            'ὁ', 'ὅς', 'καί', 'αβγ', 'ἐγώ', 'δέ', 'οὗτος', 'ἐμός'],
        stopword_basis='texts', score_basis='stem',
        freq_basis='texts', max_distance=10,
        distance_basis='span', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, results_id, texts[0].path,
                                   'mini_greek_results.tab')


def _load_v3_mini_corpus_stem_freqs(conn, language, lang_path):
    lang_lookup = {'latin': 'la.mini', 'greek': 'grc.mini'}
    freqs_path = lang_path.joinpath(
            lang_lookup[language]+'.stem.freq')
    db_cursor = conn.connection[Feature.collection].find(
            {'feature': 'lemmata', 'language': language},
            {'_id': False, 'index': True, 'token': True})
    token2index = {e['token']: e['index'] for e in db_cursor}
    freqs = {}
    with open(freqs_path, 'r', encoding='utf-8') as ifh:
        for line in ifh:
            if line.startswith('# count:'):
                denom = int(line.split()[-1])
                break
        for line in ifh:
            line = line.strip()
            if line:
                word, count = line.split()
                freqs[token2index[word]] = float(count) / denom
    return freqs


def test_mini_corpus_frequencies(
        minipop, tessfiles_greek_path,
        tessfiles_latin_path):
    for lang, lang_path in zip(
            ['greek', 'latin'],
            [tessfiles_greek_path, tessfiles_latin_path]):
        v3freqs = _load_v3_mini_corpus_stem_freqs(minipop, lang, lang_path)
        v5freqs = get_corpus_frequencies(minipop, 'lemmata', lang)
        db_cursor = minipop.connection[Feature.collection].find(
                {'feature': 'lemmata', 'language': lang},
                {'_id': False, 'index': True, 'token': True})
        index2token = {e['index']: e['token'] for e in db_cursor}
        for form_index, freq in enumerate(v5freqs):
            assert form_index in v3freqs
            assert math.isclose(v3freqs[form_index], freq), \
                f'Mismatch on {index2token[form_index]} ({form_index})'


def test_mini_latin_search_corpus_freqs(minipop, mini_latin_metadata,
                                        v3checker):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'line'),
        TextOptions(texts[1], 'line'),
        'lemmata',
        stopwords=4,
        stopword_basis='corpus', score_basis='stem',
        freq_basis='corpus', max_distance=10,
        distance_basis='frequency', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, results_id, texts[0].path,
                                   'mini_latin_corpus_results.tab')


def test_mini_greek_search_corpus_freqs(minipop, mini_greek_metadata,
                                        v3checker):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_greek_metadata])
    results_id = uuid.uuid4()
    search_result = Search(
        results_id=results_id,
        status=Search.INIT,
        msg='',
        # see tesserae.utils.search for how to actually set up Search
    )
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'phrase'),
        TextOptions(texts[1], 'phrase'),
        'lemmata',
        stopwords=10,
        stopword_basis='corpus', score_basis='stem',
        freq_basis='corpus', max_distance=10,
        distance_basis='span', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, results_id, texts[0].path,
                                   'mini_greek_corpus_results.tab')


def test_mini_punctuation(punctpop, mini_punctuation_metadata):
    texts = punctpop.find(
        Text.collection,
        title=[m['title'] for m in mini_punctuation_metadata])
    results_id = uuid.uuid4()
    search_result = Search(
        results_id=results_id,
        status=Search.INIT,
        msg='',
        # see tesserae.utils.search for how to actually set up Search
    )
    punctpop.insert(search_result)
    matcher = SparseMatrixSearch(punctpop)
    matcher.match(
        search_result,
        TextOptions(texts[0], 'phrase'),
        TextOptions(texts[1], 'phrase'),
        'lemmata',
        stopwords=10,
        stopword_basis='corpus', score_basis='stem',
        freq_basis='corpus', max_distance=10,
        distance_basis='span', min_score=0)
    # the point of this test is to make sure no Exception is thrown


def test_latin_sound(minipop, mini_latin_metadata, v3checker):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'line'),
        TextOptions(texts[1], 'line'),
        'sound',
        stopwords=['que', 'tum', 'ere'],
        stopword_basis='texts', score_basis='3gr',
        freq_basis='texts', max_distance=999,
        distance_basis='frequency', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, results_id, texts[0].path,
                                   'mini_latin_results_3gr.tab')


def test_latin_semantic(minipop, mini_latin_metadata, v3checker):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'line'),
        TextOptions(texts[1], 'line'),
        'semantic',
        stopwords=['et', 'non', 'atqui'],
        stopword_basis='texts', score_basis='stem',
        freq_basis='texts', max_distance=999,
        distance_basis='frequency', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, results_id, texts[0].path,
                                   'mini_latin_results_syn.tab')


def test_latin_semlem(minipop, mini_latin_metadata, v3checker):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'line'),
        TextOptions(texts[1], 'line'),
        'semantic + lemma',
        stopwords=['et', 'non', 'atqui'],
        stopword_basis='texts', score_basis='stem',
        freq_basis='texts', max_distance=999,
        distance_basis='frequency', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, results_id, texts[0].path,
                                   'mini_latin_results_syn_lem.tab')


def test_greek_sound(minipop, mini_greek_metadata, v3checker):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_greek_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'phrase'),
        TextOptions(texts[1], 'phrase'),
        'sound',
        stopwords=[
            'και', 'του', 'αλλ', 'ειν', 'μεν', 'μοι', 'αυτ', 'ους'],
        stopword_basis='texts', score_basis='3gr',
        freq_basis='texts', max_distance=999,
        distance_basis='span', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, results_id, texts[0].path,
                                   'mini_greek_results_3gr.tab')


def test_greek_semantic(minipop, mini_greek_metadata, v3checker):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_greek_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'phrase'),
        TextOptions(texts[1], 'phrase'),
        'semantic',
        stopwords=[
            'τις', 'οὗτος', 'καί', 'αβγ', 'ἐγώ', 'τηνόθι', 'τηνικαῦτα',
            'τέκνον'],
        stopword_basis='texts', score_basis='stem',
        freq_basis='texts', max_distance=999,
        distance_basis='span', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, results_id, texts[0].path,
                                   'mini_greek_results_syn.tab')


def test_greek_semlem(minipop, mini_greek_metadata, v3checker):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_greek_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'phrase'),
        TextOptions(texts[1], 'phrase'),
        'semantic + lemma',
        stopwords=[
            'οὗτος', 'τις', 'ὁ', 'ὅς', 'καί', 'αβγ', 'ἐγώ', 'τέκνον'],
        stopword_basis='texts', score_basis='stem',
        freq_basis='texts', max_distance=999,
        distance_basis='span', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, results_id, texts[0].path,
                                   'mini_greek_results_syn_lem.tab')


@pytest.fixture(scope='session')
def lucvergpop(request, lucverg_metadata, v3checker):
    conn = TessMongoConnection('localhost', 27017, None, None, 'lucvergtest')
    for metadata in lucverg_metadata:
        text = Text.json_decode(metadata)
        tessfile = TessFile(text.path, metadata=text)

        conn.insert(text)

        tokens, tags, features = \
            LatinTokenizer(conn).tokenize(
                tessfile.read(), text=tessfile.metadata)

        feature_cache = {(f.feature, f.token): f for f in conn.find(
            Feature.collection, language=text.language)}
        features_for_insert = []
        features_for_update = []

        for f in features:
            if (f.feature, f.token) not in feature_cache:
                features_for_insert.append(f)
                feature_cache[(f.feature, f.token)] = f
            else:
                f.id = feature_cache[(f.feature, f.token)].id
                features_for_update.append(f)
        conn.insert(features_for_insert)
        conn.update(features_for_update)

        unitizer = Unitizer()
        lines, _ = unitizer.unitize(
            tokens, tags, tessfile.metadata)

        conn.insert_nocheck(lines)
    yield conn
    obliterate(conn)


def test_lucverg(lucvergpop, lucverg_metadata, v3checker):
    texts = lucvergpop.find(
        Text.collection,
        title=[m['title'] for m in lucverg_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    lucvergpop.insert(search_result)
    matcher = SparseMatrixSearch(lucvergpop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'line'),
        TextOptions(texts[1], 'line'),
        'lemmata',
        stopwords=[
            "et", "qui", "quis", "in", "sum",
            "hic", "non", "tu", "neque", "ego"
        ],
        stopword_basis='texts', score_basis='stem',
        freq_basis='texts', max_distance=10,
        distance_basis='frequency', min_score=0)
    lucvergpop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    lucvergpop.update(search_result)
    v3checker.check_search_results(lucvergpop, results_id, texts[0].path,
                                   'lucverg_time.tab')
