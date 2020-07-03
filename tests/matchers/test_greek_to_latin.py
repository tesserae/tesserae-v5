import math
import uuid

import pytest

from tesserae.data import load_greek_to_latin
from tesserae.db import Feature, Search, Text, \
                        TessMongoConnection
from tesserae.matchers import GreekToLatinSearch
from tesserae.matchers.greek_to_latin import \
    _build_greekind_to_latininds, _get_greek_to_latin_inv_freqs_by_text
from tesserae.matchers.sparse_encoding import _get_units
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


def test_greek_to_latin_inv_freq_by_text(g2lpop, v3checker):
    greekind_to_latininds = _build_greekind_to_latininds(
        g2lpop, load_greek_to_latin())
    greek_text = g2lpop.find(Text.collection, language='greek')[0]
    greek_text_options = TextOptions(greek_text, 'line')
    greek_text_length = sum(
        len(u['forms'])
        for u in _get_units(g2lpop, greek_text_options, 'lemmata'))
    inv_freqs = _get_greek_to_latin_inv_freqs_by_text(g2lpop,
                                                      greek_text_options,
                                                      greek_text_length,
                                                      greekind_to_latininds)
    v3_total, v3_counts = v3checker._load_v3_mini_text_freqs_file(
        g2lpop, greek_text, 'g_l')
    assert len(v3_counts) == len(inv_freqs)
    greek_forms = {
        f.index: f.token
        for f in g2lpop.find(
            Feature.collection, language='greek', feature='form')
    }
    for token, count in v3_counts.items():
        assert token in inv_freqs
        assert math.isclose(inv_freqs[token],
                            float(v3_total) / count), greek_forms[token]


def test_greek_to_latin(g2lpop, mini_g2l_metadata, v3checker):
    texts = g2lpop.find(Text.collection,
                        title=[m['title'] for m in mini_g2l_metadata])
    results_id = uuid.uuid4()
    search_result = Search(results_id=results_id)
    g2lpop.insert(search_result)
    matcher = GreekToLatinSearch(g2lpop)
    v5_matches = matcher.match(search_result,
                               TextOptions(texts[0], 'line'),
                               TextOptions(texts[1], 'line'),
                               greek_stopwords=[],
                               latin_stopwords=['is', 'quis', 'atque'],
                               freq_basis='texts',
                               max_distance=999,
                               distance_basis='frequency',
                               min_score=0)
    g2lpop.insert_nocheck(v5_matches)
    search_result.status = Search.DONE
    g2lpop.update(search_result)
    v3checker.check_search_results(g2lpop, results_id, texts[0].path,
                                   'mini_g2l.tab')
