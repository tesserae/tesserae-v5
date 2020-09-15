from tesserae.db.entities import Feature, Text
from tesserae.utils.ingest import add_feature


def test_add_feature(minipop):
    texts = minipop.find(Text.collection)
    feature = 'test'
    for text in texts:
        add_feature(minipop, text, feature)
        text_id_str = str(text.id)
        db_test_features = {
            f.token: f.frequencies[text_id_str]
            for f in minipop.find(
                Feature.collection, language=text.language, feature=feature)
        }
        # test feature is equivalent to sound feature
        db_sound_features = {
            f.token: f.frequencies[text_id_str]
            for f in minipop.find(
                Feature.collection, language=text.language, feature='sound')
            if text_id_str in f.frequencies
        }
        tokens_in_test_not_in_sound = [
            token for token in db_test_features
            if token not in db_sound_features
        ]
        assert not tokens_in_test_not_in_sound
        discrepancies = []
        for token, freq in db_test_features.items():
            if freq != db_sound_features[token]:
                discrepancies.append((token, freq, db_sound_features[token]))
        assert not discrepancies
