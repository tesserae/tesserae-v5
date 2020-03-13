from tesserae.db.entities import Feature, Text
from tesserae.utils.search import bigram_search


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
    units = bigram_search(
        minipop, bellum.index, pando.index, feature, 'line',
        [t.id for t in texts])
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
