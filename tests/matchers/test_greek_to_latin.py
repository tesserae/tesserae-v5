import uuid

import pytest

from tesserae.db import Search, Text, \
                        TessMongoConnection
from tesserae.matchers import GreekToLatinSearch
from tesserae.matchers.text_options import TextOptions
from tesserae.utils import ingest_text
from tesserae.utils.delete import obliterate


@pytest.fixture(scope='session')
def g2lpop(request, mini_g2l_metadata):
    conn = TessMongoConnection('localhost', 27017, None, None, 'g2ltest')
    for metadata in mini_g2l_metadata:
        text = Text.json_decode(metadata)
        ingest_text(conn, text)
    yield conn
    obliterate(conn)


def test_greek_to_latin(g2lpop, mini_g2l_metadata, v3checker):
    texts = g2lpop.find(
        Text.collection,
        title=[m['title'] for m in mini_g2l_metadata]
    )
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    g2lpop.insert(search_result)
    matcher = GreekToLatinSearch(g2lpop)
    v5_matches = matcher.match(
        search_result,
        TextOptions(texts[0], 'line'),
        TextOptions(texts[1], 'line'),
        greek_stopwords=[],
        latin_stopwords=['is', 'quis', 'atque'],
        freq_basis='texts',
        max_distance=999,
        distance_basis='frequency',
        min_score=0
    )
    g2lpop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    g2lpop.update(search_result)
    v3checker.check_search_results(g2lpop, results_id, texts[0].path,
                                   'mini_g2l.tab')
