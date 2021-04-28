import uuid

import numpy as np
import pytest
from tesserae.db import Feature, Search, TessMongoConnection, Text
from tesserae.matchers.sparse_encoding import SparseMatrixSearch, _get_units
from tesserae.matchers.text_options import TextOptions
from tesserae.tokenizers import LatinTokenizer
from tesserae.unitizer import Unitizer
from tesserae.utils import ingest_text
from tesserae.utils.delete import obliterate
from tesserae.utils.stopwords import get_stoplist_tokens
from tesserae.utils.tessfile import TessFile
from tests.conftest import _load_v3_results


@pytest.fixture(scope='session')
def punctpop(request, mini_punctuation_metadata):
    conn = TessMongoConnection('localhost', 27017, None, None, 'puncttess')
    for metadata in mini_punctuation_metadata:
        text = Text.json_decode(metadata)
        ingest_text(conn, text)
    yield conn
    obliterate(conn)


def test_mini_latin_search_text_freqs(minipop, mini_latin_metadata, v3checker):
    texts = minipop.find(Text.collection,
                         title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'line'),
                               TextOptions(texts[1], 'line'),
                               'lemmata',
                               stopwords=['et', 'neque', 'qui'],
                               stopword_basis='texts',
                               score_basis='lemmata',
                               freq_basis='texts',
                               max_distance=10,
                               distance_basis='frequency',
                               min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, search_result.id, texts[0].path,
                                   'mini_latin_results.tab')


def test_mini_greek_search_text_freqs(minipop, mini_greek_metadata, v3checker):
    texts = minipop.find(Text.collection,
                         title=[m['title'] for m in mini_greek_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'phrase'),
                               TextOptions(texts[1], 'phrase'),
                               'lemmata',
                               stopwords=[
                                   'ὁ', 'ὅς', 'καί', 'αβγ', 'ἐγώ', 'δέ',
                                   'οὗτος', 'ἐμός'
                               ],
                               stopword_basis='texts',
                               score_basis='lemmata',
                               freq_basis='texts',
                               max_distance=10,
                               distance_basis='span',
                               min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, search_result.id, texts[0].path,
                                   'mini_greek_results.tab')


def test_mini_latin_search_corpus_freqs(minipop, mini_latin_metadata,
                                        v3checker):
    texts = minipop.find(Text.collection,
                         title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'line'),
                               TextOptions(texts[1], 'line'),
                               'lemmata',
                               stopwords=4,
                               stopword_basis='corpus',
                               score_basis='lemmata',
                               freq_basis='corpus',
                               max_distance=10,
                               distance_basis='frequency',
                               min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, search_result.id, texts[0].path,
                                   'mini_latin_corpus_results.tab')


def test_mini_greek_search_corpus_freqs(minipop, mini_greek_metadata,
                                        v3checker):
    texts = minipop.find(Text.collection,
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
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'phrase'),
                               TextOptions(texts[1], 'phrase'),
                               'lemmata',
                               stopwords=10,
                               stopword_basis='corpus',
                               score_basis='lemmata',
                               freq_basis='corpus',
                               max_distance=10,
                               distance_basis='span',
                               min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, search_result.id, texts[0].path,
                                   'mini_greek_corpus_results.tab')


def test_mini_punctuation(punctpop, mini_punctuation_metadata):
    texts = punctpop.find(
        Text.collection, title=[m['title'] for m in mini_punctuation_metadata])
    results_id = uuid.uuid4()
    search_result = Search(
        results_id=results_id,
        status=Search.INIT,
        msg='',
        # see tesserae.utils.search for how to actually set up Search
    )
    punctpop.insert(search_result)
    matcher = SparseMatrixSearch(punctpop)
    matcher.match(search_result,
                  TextOptions(texts[0], 'phrase'),
                  TextOptions(texts[1], 'phrase'),
                  'lemmata',
                  stopwords=10,
                  stopword_basis='corpus',
                  score_basis='lemmata',
                  freq_basis='corpus',
                  max_distance=10,
                  distance_basis='span',
                  min_score=0)
    # the point of this test is to make sure no Exception is thrown


def test_latin_sound(minipop, mini_latin_metadata, v3checker):
    texts = minipop.find(Text.collection,
                         title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'line'),
                               TextOptions(texts[1], 'line'),
                               'sound',
                               stopwords=['que', 'tum', 'ere'],
                               stopword_basis='texts',
                               score_basis='sound',
                               freq_basis='texts',
                               max_distance=999,
                               distance_basis='frequency',
                               min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, search_result.id, texts[0].path,
                                   'mini_latin_results_3gr.tab')


def test_latin_trigrams(minipop, mini_latin_metadata):
    """
    For the purpose of visualization.
    Use to confirm that trigrams are being stored in the database correctly.
    It should be noted that v5 results do not have stopwords filtered out,
    while v3 results probably do.
    """
    texts = minipop.find(Text.collection,
                         title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    v5_results = []
    v3_results = []
    raw_v5_results = []
    target_units = _get_units(minipop, TextOptions(texts[0], 'line'), 'sound')
    for b in target_units:
        raw_v5_results.append(b['features'])
    raw_v3_results = _load_v3_results(texts[0].path,
                                      'mini_latin_results_3gr.tab')
    for a in raw_v3_results:
        v3_results.append(a['matched_features'])
    print('v5 results:')
    for a in raw_v5_results:
        print(a)
        for n in a:
            #            print(n)
            n = np.asarray(n)
            #            print('array',n)
            #            print('shape', np.shape(n))
            b = get_stoplist_tokens(minipop, n, 'sound', 'latin')
            v5_results.append(b)
    print(v5_results)
    print('v3 results:')
    for a in v3_results:
        print(a)
    print('v5 length:', len(v5_results), 'v3 length:', len(v3_results))
    assert False


def test_latin_semantic(minipop, mini_latin_metadata, v3checker):
    texts = minipop.find(Text.collection,
                         title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'line'),
                               TextOptions(texts[1], 'line'),
                               'semantic',
                               stopwords=['et', 'non', 'atqui'],
                               stopword_basis='texts',
                               score_basis='lemmata',
                               freq_basis='texts',
                               max_distance=999,
                               distance_basis='frequency',
                               min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, search_result.id, texts[0].path,
                                   'mini_latin_results_syn.tab')


def test_latin_semlem(minipop, mini_latin_metadata, v3checker):
    texts = minipop.find(Text.collection,
                         title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'line'),
                               TextOptions(texts[1], 'line'),
                               'semantic + lemmata',
                               stopwords=['et', 'non', 'neque'],
                               stopword_basis='texts',
                               score_basis='lemmata',
                               freq_basis='texts',
                               max_distance=999,
                               distance_basis='frequency',
                               min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, search_result.id, texts[0].path,
                                   'mini_latin_results_syn_lem.tab')


def test_greek_sound(minipop, mini_greek_metadata, v3checker):
    texts = minipop.find(Text.collection,
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
        stopwords=['και', 'του', 'αλλ', 'ειν', 'μεν', 'μοι', 'αυτ', 'ους'],
        stopword_basis='texts',
        score_basis='sound',
        freq_basis='texts',
        max_distance=999,
        distance_basis='span',
        min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, search_result.id, texts[0].path,
                                   'mini_greek_results_3gr.tab')


def test_greek_trigrams(minipop, mini_greek_metadata):
    """
    For the purpose of visualization.
    Use to confirm that trigrams are being stored in the database correctly.
    It should be noted that v5 results do not have stopwords filtered out,
    while v3 results probably do.
    """
    texts = minipop.find(Text.collection,
                         title=[m['title'] for m in mini_greek_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    v5_results = []
    v3_results = []
    raw_v5_results = []
    target_units = _get_units(minipop, TextOptions(texts[0], 'line'), 'sound')
    for b in target_units:
        raw_v5_results.append(b['features'])
    raw_v3_results = _load_v3_results(texts[0].path,
                                      'mini_greek_results_3gr.tab')
    for a in raw_v3_results:
        v3_results.append(a['matched_features'])
    print('v5 results:')
    for a in raw_v5_results:
        print(a)
        for n in a:
            #            print(n)
            n = np.asarray(n)
            #            print('array',n)
            #            print('shape', np.shape(n))
            b = get_stoplist_tokens(minipop, n, 'sound', 'greek')
            v5_results.append(b)
    print(v5_results)
    print('v3 results:')
    for a in v3_results:
        print(a)
    print('v5 length:', len(v5_results), 'v3 length:', len(v3_results))
    assert False


def test_greek_semantic(minipop, mini_greek_metadata, v3checker):
    texts = minipop.find(Text.collection,
                         title=[m['title'] for m in mini_greek_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'phrase'),
                               TextOptions(texts[1], 'phrase'),
                               'semantic',
                               stopwords=[
                                   'τις', 'οὗτος', 'καί', 'αβγ', 'ἐγώ',
                                   'τηνόθι', 'τηνικαῦτα', 'τέκνον'
                               ],
                               stopword_basis='texts',
                               score_basis='lemmata',
                               freq_basis='texts',
                               max_distance=999,
                               distance_basis='span',
                               min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, search_result.id, texts[0].path,
                                   'mini_greek_results_syn.tab')


def test_greek_semlem(minipop, mini_greek_metadata, v3checker):
    texts = minipop.find(Text.collection,
                         title=[m['title'] for m in mini_greek_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'phrase'),
                               TextOptions(texts[1], 'phrase'),
                               'semantic + lemmata',
                               stopwords=[
                                   'οὗτος', 'τις', 'ὁ', 'ὅς', 'καί',
                                   'αβγ', 'ἐγώ', 'τέκνον'
                               ],
                               stopword_basis='texts',
                               score_basis='lemmata',
                               freq_basis='texts',
                               max_distance=999,
                               distance_basis='span',
                               min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v3checker.check_search_results(minipop, search_result.id, texts[0].path,
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

        feature_cache = {
            (f.feature, f.token): f
            for f in conn.find(Feature.collection, language=text.language)
        }
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
        lines, _ = unitizer.unitize(tokens, tags, tessfile.metadata)

        conn.insert_nocheck(lines)
    yield conn
    obliterate(conn)


def test_lucverg(lucvergpop, lucverg_metadata, v3checker):
    texts = lucvergpop.find(Text.collection,
                            title=[m['title'] for m in lucverg_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    lucvergpop.insert(search_result)
    matcher = SparseMatrixSearch(lucvergpop)
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'line'),
                               TextOptions(texts[1], 'line'),
                               'lemmata',
                               stopwords=[
                                   "et", "qui", "quis", "in", "sum", "hic",
                                   "non", "tu", "neque", "ego"
                               ],
                               stopword_basis='texts',
                               score_basis='lemmata',
                               freq_basis='texts',
                               max_distance=10,
                               distance_basis='frequency',
                               min_score=0)
    lucvergpop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    lucvergpop.update(search_result)
    v3checker.check_search_results(lucvergpop, search_result.id, texts[0].path,
                                   'lucverg_time.tab')


@pytest.fixture(scope='session')
def engpop(request, eng_metadata, v3checker):
    conn = TessMongoConnection('localhost', 27017, None, None, 'engtest')
    for metadata in eng_metadata:
        text = Text.json_decode(metadata)
        ingest_text(conn, text)
    yield conn
    obliterate(conn)


def test_english(engpop, eng_metadata, v3checker):
    texts = engpop.find(Text.collection,
                        title=[m['title'] for m in eng_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    engpop.insert(search_result)
    matcher = SparseMatrixSearch(engpop)
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'line'),
                               TextOptions(texts[1], 'line'),
                               'form',
                               stopwords=[
                                   "the",
                                   "and",
                                   "of",
                                   "a",
                                   "to",
                                   "in",
                                   "that",
                                   "with",
                                   "i",
                                   "by",
                               ],
                               stopword_basis='texts',
                               score_basis='form',
                               freq_basis='texts',
                               max_distance=10,
                               distance_basis='frequency',
                               min_score=6.0)
    engpop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    engpop.update(search_result)
    v3checker.check_search_results(engpop, search_result.id, texts[0].path,
                                   'eng_time.tab')
