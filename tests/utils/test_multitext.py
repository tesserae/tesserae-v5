import itertools
import uuid

from tesserae.db.entities import Search, Text
from tesserae.matchers.sparse_encoding import SparseMatrixSearch
from tesserae.matchers.text_options import TextOptions
from tesserae.utils.multitext import multitext_search


def test_latin_multitext_search(minipop):
    feature = 'lemmata'
    language = 'latin'
    texts = minipop.find(Text.collection, language=language)

    results_id = uuid.uuid4()
    search_result = Search(
        results_id=results_id,
        status=Search.INIT,
        msg='',
        # see tesserae.utils.search for how to actually set up Search
    )
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    matches = matcher.match(search_result,
                            TextOptions(texts[0], 'line'),
                            TextOptions(texts[1], 'line'),
                            'lemmata',
                            stopwords=['et', 'neque', 'qui'],
                            stopword_basis='corpus',
                            score_basis='lemmata',
                            freq_basis='corpus',
                            max_distance=10,
                            distance_basis='span',
                            min_score=0)

    results = multitext_search(search_result, minipop, matches, feature,
                               'line', texts)
    assert len(results) == len(matches)
    for r, m in zip(results, matches):
        bigrams = [
            bigram
            for bigram in itertools.combinations(sorted(m.matched_features), 2)
        ]
        assert len(bigrams) == len(r)
        for bigram in bigrams:
            assert bigram in r


def test_greek_multitext_search(minipop):
    feature = 'lemmata'
    language = 'greek'
    texts = minipop.find(Text.collection, language=language)

    results_id = uuid.uuid4()
    search_result = Search(
        results_id=results_id,
        status=Search.INIT,
        msg='',
        # see tesserae.utils.search for how to actually set up Search
    )
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    matches = matcher.match(search_result,
                            TextOptions(texts[0], 'line'),
                            TextOptions(texts[1], 'line'),
                            'lemmata',
                            stopwords=[
                                'ὁ', 'ὅς', 'καί', 'αβγ', 'ἐγώ', 'δέ',
                                'οὗτος', 'ἐμός'
                            ],
                            stopword_basis='corpus',
                            score_basis='lemmata',
                            freq_basis='corpus',
                            max_distance=10,
                            distance_basis='span',
                            min_score=0)

    results = multitext_search(search_result, minipop, matches, feature,
                               'line', texts)
    assert len(results) == len(matches)
    for r, m in zip(results, matches):
        bigrams = [
            bigram
            for bigram in itertools.combinations(sorted(m.matched_features), 2)
        ]
        assert len(bigrams) == len(r)
        for bigram in bigrams:
            assert bigram in r
