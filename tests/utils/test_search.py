import pytest
import random
import string
import uuid

from tesserae.db import TessMongoConnection
from tesserae.db.entities import Search, Match
from tesserae.utils.search import get_results, NORMAL_SEARCH, PageOptions


def _create_random_word():
    LONGEST_WORD = 50
    word_length = random.randint(1, LONGEST_WORD)
    # https://stackoverflow.com/questions/2823316/generate-a-random-letter-in-python
    return ''.join(random.choices(string.ascii_letters, k=word_length))


def _create_match(search):
    ARBITRARY_NUMBER = 9000
    source_book = random.randrange(ARBITRARY_NUMBER)
    source_section = random.randrange(ARBITRARY_NUMBER)
    target_book = random.randrange(ARBITRARY_NUMBER)
    target_section = random.randrange(ARBITRARY_NUMBER)
    score = random.random()
    return Match(
        search_id=search.id,
        source_tag=f'test {source_book}.{source_section}',
        target_tag=f'test {target_book}.{target_section}',
        matched_features=[_create_random_word(), _create_random_word()],
        score=score
    )


@pytest.fixture(scope='session')
def resultsdb():
    conn = TessMongoConnection('localhost', 27017, None, None, 'resultdb')
    results_id = uuid.uuid4()
    search_result = Search(
        results_id=results_id,
        search_type=NORMAL_SEARCH,
        status=Search.DONE
    )
    conn.insert(search_result)
    conn.insert([_create_match(search_result) for _ in range(100)])
    yield conn
    for coll_name in conn.connection.list_collection_names():
        conn.connection.drop_collection(coll_name)


def _tuplize_match(match):
    return (match.source_tag, match.target_tag, tuple(match.matched_features),
            match.score)


def _tuplize_match_result(match_result):
    return (match_result['source_tag'], match_result['target_tag'],
            tuple(match_result['matched_features']), match_result['score'])


def _assert_equivalent_results(got_results, true_results):
    got_set = set([
        _tuplize_match_result(r)
        for r in got_results
    ])
    true_set = set([
        _tuplize_match(t)
        for t in true_results
    ])
    assert len(got_set) == len(true_set)
    assert got_set <= true_set and true_set <= got_set


def test_get_results_dump(resultsdb):
    search = resultsdb.find(Search.collection)[0]
    got_results = get_results(resultsdb, search.id, PageOptions())
    true_results = resultsdb.find(Match.collection, search_id=search.id)
    _assert_equivalent_results(got_results, true_results)


def test_get_results_sort_score(resultsdb):
    search = resultsdb.find(Search.collection)[0]
    page_options = PageOptions(
        sort_by='score',
        sort_order='descending',
        per_page=20,
        page_number=0
    )
    got_results = get_results(resultsdb, search.id, page_options)
    true_results = resultsdb.find(Match.collection, search_id=search.id)
    true_results.sort(key=lambda x: x.score, reverse=True)
    _assert_equivalent_results(got_results, true_results[0:20])

    page_options = PageOptions(
        sort_by='score',
        sort_order='descending',
        per_page=50,
        page_number=1
    )
    got_results = get_results(resultsdb, search.id, page_options)
    _assert_equivalent_results(got_results, true_results[50:100])

    page_options = PageOptions(
        sort_by='score',
        sort_order='ascending',
        per_page=20,
        page_number=2
    )
    got_results = get_results(resultsdb, search.id, page_options)
    true_results.sort(key=lambda x: x.score, reverse=False)
    _assert_equivalent_results(got_results, true_results[40:60])
    page_options.sort_order = 1


def test_get_results_sort_source_tag(resultsdb):
    search = resultsdb.find(Search.collection)[0]
    page_options = PageOptions(
        sort_by='source_tag',
        sort_order='descending',
        per_page=20,
        page_number=0
    )
    got_results = get_results(resultsdb, search.id, page_options)
    true_results = resultsdb.find(Match.collection, search_id=search.id)
    true_results.sort(key=lambda x: x.source_tag, reverse=True)
    _assert_equivalent_results(got_results, true_results[0:20])

    page_options = PageOptions(
        sort_by='source_tag',
        sort_order='descending',
        per_page=50,
        page_number=1
    )
    got_results = get_results(resultsdb, search.id, page_options)
    _assert_equivalent_results(got_results, true_results[50:100])

    page_options = PageOptions(
        sort_by='source_tag',
        sort_order='ascending',
        per_page=20,
        page_number=2
    )
    got_results = get_results(resultsdb, search.id, page_options)
    true_results.sort(key=lambda x: x.source_tag, reverse=False)
    _assert_equivalent_results(got_results, true_results[40:60])
    page_options.sort_order = 1


def test_get_results_sort_target_tag(resultsdb):
    search = resultsdb.find(Search.collection)[0]
    page_options = PageOptions(
        sort_by='target_tag',
        sort_order='descending',
        per_page=20,
        page_number=0
    )
    got_results = get_results(resultsdb, search.id, page_options)
    true_results = resultsdb.find(Match.collection, search_id=search.id)
    true_results.sort(key=lambda x: x.target_tag, reverse=True)
    _assert_equivalent_results(got_results, true_results[0:20])

    page_options = PageOptions(
        sort_by='target_tag',
        sort_order='descending',
        per_page=50,
        page_number=1
    )
    got_results = get_results(resultsdb, search.id, page_options)
    _assert_equivalent_results(got_results, true_results[50:100])

    page_options = PageOptions(
        sort_by='target_tag',
        sort_order='ascending',
        per_page=20,
        page_number=2
    )
    got_results = get_results(resultsdb, search.id, page_options)
    true_results.sort(key=lambda x: x.target_tag, reverse=False)
    _assert_equivalent_results(got_results, true_results[40:60])
    page_options.sort_order = 1


def test_get_results_sort_matched_features(resultsdb):
    search = resultsdb.find(Search.collection)[0]
    page_options = PageOptions(
        sort_by='matched_features',
        sort_order='descending',
        per_page=20,
        page_number=0
    )
    got_results = get_results(resultsdb, search.id, page_options)
    true_results = resultsdb.find(Match.collection, search_id=search.id)
    true_results.sort(key=lambda x: x.matched_features, reverse=True)
    _assert_equivalent_results(got_results, true_results[0:20])

    page_options = PageOptions(
        sort_by='matched_features',
        sort_order='descending',
        per_page=50,
        page_number=1
    )
    got_results = get_results(resultsdb, search.id, page_options)
    _assert_equivalent_results(got_results, true_results[50:100])

    page_options = PageOptions(
        sort_by='matched_features',
        sort_order='ascending',
        per_page=20,
        page_number=2
    )
    got_results = get_results(resultsdb, search.id, page_options)
    true_results.sort(key=lambda x: x.matched_features, reverse=False)
    _assert_equivalent_results(got_results, true_results[40:60])
    page_options.sort_order = 1
