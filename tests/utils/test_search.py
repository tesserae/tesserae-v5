import itertools
import uuid

from tesserae.db.entities import Feature, Search, Text
from tesserae.matchers.sparse_encoding import SparseMatrixSearch
from tesserae.matchers.text_options import TextOptions
from tesserae.utils.search import bigram_search, multitext_search


def test_bigram_search(minipop, mini_latin_metadata):
    feature = 'lemmata'
    language = 'latin'
    bellum = minipop.find(
        Feature.collection, language=language, token='bellum',
        feature=feature)[0]
    pando = minipop.find(
        Feature.collection, language=language, token='pando',
        feature=feature)[0]
    texts = minipop.find(
        Text.collection, language=language
    )
    units = []
    for t in texts:
        units.extend(bigram_search(
            minipop, bellum.index, pando.index, feature, 'line', t.id))
    assert len(units) > 0
    for u in units:
        bellum_found = False
        pando_found = False
        for t in u.tokens:
            cur_features = t['features'][feature]
            if bellum.index in cur_features and \
                    pando.index not in cur_features:
                bellum_found = True
            if pando.index in cur_features and \
                    bellum.index not in cur_features:
                pando_found = True
        assert bellum_found
        assert pando_found


def test_multitext_search(minipop, mini_latin_metadata):
    feature = 'lemmata'
    language = 'latin'
    texts = minipop.find(
        Text.collection, language=language
    )

    results_id = uuid.uuid4()
    search_result = Search(
        results_id=results_id,
        status=Search.INIT,
        msg='',
        # see tesserae.utils.search for how to actually set up Search
    )
    minipop.insert(search_result)
    matcher = SparseMatrixSearch(minipop)
    matches = matcher.match(
        search_result.id,
        TextOptions(texts[0], 'line'),
        TextOptions(texts[1], 'line'),
        'lemmata',
        stopwords=['et', 'neque', 'qui'],
        stopword_basis='corpus', score_basis='stem',
        frequency_basis='corpus', max_distance=10,
        distance_metric='span', min_score=0)

    results = multitext_search(minipop, matches, feature, 'line', texts)
    assert len(results) == len(matches)
    for r, m in zip(results, matches):
        bigrams = [
            bigram
            for bigram in itertools.combinations(sorted(m.matched_features), 2)
        ]
        assert len(bigrams) == len(r)
        for bigram in bigrams:
            assert bigram in r
