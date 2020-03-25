import itertools
from pathlib import Path
import pprint
import uuid

import math
import pytest

from tesserae.db import Feature, Search, Text, \
                        TessMongoConnection
from tesserae.matchers.sparse_encoding import \
        SparseMatrixSearch, get_text_frequencies, get_corpus_frequencies
from tesserae.matchers.text_options import TextOptions
from tesserae.utils import ingest_text
from tesserae.utils.retrieve import get_results


@pytest.fixture(scope='session')
def punctpop(request, mini_punctuation_metadata):
    conn = TessMongoConnection('localhost', 27017, None, None, 'puncttess')
    for metadata in mini_punctuation_metadata:
        text = Text.json_decode(metadata)
        ingest_text(conn, text)
    yield conn
    for coll_name in conn.connection.list_collection_names():
        conn.connection.drop_collection(coll_name)


def _load_v3_mini_text_stem_freqs(conn, metadata):
    db_cursor = conn.connection[Feature.collection].find(
            {'feature': 'form', 'language': metadata['language']},
            {'_id': False, 'index': True, 'token': True})
    token2index = {e['token']: e['index'] for e in db_cursor}
    # the .freq_score_stem file is named the same as its corresponding .tess
    # file
    freqs_path = metadata['path'][:-4] + 'freq_score_stem'
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


def test_mini_text_frequencies(
        minipop, mini_latin_metadata,
        mini_greek_metadata):
    all_text_metadata = [m for m in itertools.chain.from_iterable(
        [mini_latin_metadata, mini_greek_metadata])]
    title2id = {t.title: t.id for t in minipop.find(
        Text.collection, title=[m['title'] for m in all_text_metadata])}
    for metadata in all_text_metadata:
        v3freqs = _load_v3_mini_text_stem_freqs(minipop, metadata)
        text_id = title2id[metadata['title']]
        v5freqs = get_text_frequencies(minipop, 'lemmata', text_id)
        for form_index, freq in v5freqs.items():
            assert form_index in v3freqs
            assert math.isclose(v3freqs[form_index], freq)


def _load_v3_results(minitext_path, tab_filename):
    tab_filepath = Path(minitext_path).resolve().parent.joinpath(tab_filename)
    v3_results = []
    with open(tab_filepath, 'r', encoding='utf-8') as ifh:
        for line in ifh:
            if not line.startswith('#'):
                break
        for line in ifh:
            # ignore headers
            # headers = line.strip().split('\t')
            break
        for line in ifh:
            data = line.strip().split('\t')
            v3_results.append({
                'source_tag': data[3][1:-1],
                'target_tag': data[1][1:-1],
                'matched_features': data[5][1:-1].split('; '),
                'score': float(data[6]),
                'source_snippet': data[4][1:-1],
                'target_snippet': data[2][1:-1],
                'highlight': ''
            })
    return v3_results


def _build_relations(results):
    relations = {}
    for match in results:
        target_loc = match['target_tag'].split()[-1]
        source_loc = match['source_tag'].split()[-1]
        if target_loc not in relations:
            relations[target_loc] = {source_loc: match}
        elif source_loc not in relations[target_loc]:
            relations[target_loc][source_loc] = match
    return relations


def _check_search_results(v5_results, v3_results):
    v3_relations = _build_relations(v3_results)
    v5_relations = _build_relations(v5_results)
    score_discrepancies = []
    match_discrepancies = []
    in_v5_not_in_v3 = []
    in_v3_not_in_v5 = []
    for target_loc in v3_relations:
        for source_loc in v3_relations[target_loc]:
            if target_loc not in v5_relations or \
                    source_loc not in v5_relations[target_loc]:
                in_v3_not_in_v5.append(v3_relations[target_loc][source_loc])
                continue
            v3_match = v3_relations[target_loc][source_loc]
            v5_match = v5_relations[target_loc][source_loc]
            v3_score = v3_match['score']
            v5_score = v5_match['score']
            if f'{v5_score:.3f}' != f'{v3_score:.3f}':
                score_discrepancies.append((
                    target_loc, source_loc,
                    v5_score-v3_score))
            v5_match_features = set(v5_match['matched_features'])
            v3_match_features = set()
            for match_f in v3_match['matched_features']:
                for f in match_f.split('-'):
                    v3_match_features.add(f)
            only_in_v5 = v5_match_features - v3_match_features
            only_in_v3 = v3_match_features - v5_match_features
            if only_in_v5 or only_in_v3:
                match_discrepancies.append((
                    target_loc, source_loc, only_in_v5,
                    only_in_v3))
    for target_loc in v5_relations:
        for source_loc in v5_relations[target_loc]:
            if target_loc not in v3_relations or \
                    source_loc not in v3_relations[target_loc]:
                in_v5_not_in_v3.append(v5_relations[target_loc][source_loc])
    pprint.pprint(score_discrepancies)
    pprint.pprint(match_discrepancies)
    pprint.pprint(in_v5_not_in_v3)
    pprint.pprint(in_v3_not_in_v5)
    assert not score_discrepancies
    assert not match_discrepancies
    assert not in_v5_not_in_v3
    assert not in_v3_not_in_v5


def test_mini_latin_search_text_freqs(minipop, mini_latin_metadata):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result.id,
        TextOptions(texts[0], 'line'),
        TextOptions(texts[1], 'line'),
        'lemmata',
        stopwords=['et', 'neque', 'qui'],
        stopword_basis='texts', score_basis='stem',
        frequency_basis='texts', max_distance=10,
        distance_metric='frequency', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v5_results = get_results(minipop, results_id)
    v5_results = sorted(v5_results, key=lambda x: -x['score'])
    v3_results = _load_v3_results(texts[0].path, 'mini_latin_results.tab')
    _check_search_results(v5_results, v3_results)


def test_mini_greek_search_text_freqs(minipop, mini_greek_metadata):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_greek_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result.id,
        TextOptions(texts[0], 'phrase'),
        TextOptions(texts[1], 'phrase'),
        'lemmata',
        stopwords=[
            'ὁ', 'ὅς', 'καί', 'αβγ', 'ἐγώ', 'δέ', 'οὗτος', 'ἐμός'],
        stopword_basis='texts', score_basis='stem',
        frequency_basis='texts', max_distance=10,
        distance_metric='span', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v5_results = get_results(minipop, results_id)
    v5_results = sorted(v5_results, key=lambda x: -x['score'])
    v3_results = _load_v3_results(texts[0].path, 'mini_greek_results.tab')
    print(len(v5_results), len(v3_results))
    _check_search_results(v5_results, v3_results)


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


def test_mini_latin_search_corpus_freqs(minipop, mini_latin_metadata):
    texts = minipop.find(
        Text.collection,
        title=[m['title'] for m in mini_latin_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    v5_matches = matcher.match(
        search_result.id,
        TextOptions(texts[0], 'line'),
        TextOptions(texts[1], 'line'),
        'lemmata',
        stopwords=4,
        stopword_basis='corpus', score_basis='stem',
        frequency_basis='corpus', max_distance=10,
        distance_metric='frequency', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v5_results = get_results(minipop, results_id)
    v5_results = sorted(v5_results, key=lambda x: -x['score'])
    v3_results = _load_v3_results(
            texts[0].path, 'mini_latin_corpus_results.tab')
    _check_search_results(v5_results, v3_results)


def test_mini_greek_search_corpus_freqs(minipop, mini_greek_metadata):
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
        search_result.id,
        TextOptions(texts[0], 'phrase'),
        TextOptions(texts[1], 'phrase'),
        'lemmata',
        stopwords=10,
        stopword_basis='corpus', score_basis='stem',
        frequency_basis='corpus', max_distance=10,
        distance_metric='span', min_score=0)
    minipop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    minipop.update(search_result)
    v5_results = get_results(minipop, results_id)
    v5_results = sorted(v5_results, key=lambda x: -x['score'])
    v3_results = _load_v3_results(
            texts[0].path, 'mini_greek_corpus_results.tab')
    _check_search_results(v5_results, v3_results)


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
        search_result.id,
        TextOptions(texts[0], 'phrase'),
        TextOptions(texts[1], 'phrase'),
        'lemmata',
        stopwords=10,
        stopword_basis='corpus', score_basis='stem',
        frequency_basis='corpus', max_distance=10,
        distance_metric='span', min_score=0)
    # the point of this test is to make sure no Exception is thrown
