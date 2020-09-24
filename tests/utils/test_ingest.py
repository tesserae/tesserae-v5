from tesserae.db.entities import Feature, Text, Token, Unit
from tesserae.utils.ingest import add_feature


def test_add_feature(minipop):
    texts = minipop.find(Text.collection)
    feature = 'test'
    for text in texts:
        add_feature(minipop, text, feature)
        db_test_features = {
            f.token: f
            for f in minipop.find(
                Feature.collection, language=text.language, feature=feature)
        }
        # test feature is equivalent to sound feature
        db_sound_features = {
            f.token: f
            for f in minipop.find(
                Feature.collection, language=text.language, feature='sound')
        }

        # make sure tokens in test are found in sound
        tokens_in_test_not_in_sound = [
            token for token in db_test_features
            if token not in db_sound_features
        ]
        assert not tokens_in_test_not_in_sound

        # make sure frequencies for this text in test are the same in sound
        discrepancies = []
        text_id_str = str(text.id)
        for token, f in db_test_features.items():
            if text_id_str not in f.frequencies:
                continue
            freq = f.frequencies[text_id_str]
            other_f = db_sound_features[token]
            if text_id_str in other_f:
                other_freq = other_f.frequencies[text_id_str]
                if freq != other_freq:
                    discrepancies.append((token, freq, other_freq))
            else:
                discrepancies.append((token, freq, 0))
        assert not discrepancies

        # make sure both test and sound features are stored in the text's units
        test_to_sound = {
            f.index: db_sound_features[token].index
            for token, f in db_test_features.items()
        }
        discrepancies = []
        units = minipop.find(Unit.collection, text=text.id)
        for unit in units:
            for token in units.tokens:
                for index in token['features'][feature]:
                    if test_to_sound[index] not in token['features']['sound']:
                        discrepancies.append((unit, token, index))
        assert not discrepancies

        # make sure text's tokens were updated properly
        test_to_sound = {
            f.id: db_sound_features[token].id
            for token, f in db_test_features.items()
        }
        discrepancies = []
        tokens = minipop.find(Token.collection, text=text.id)
        for token in tokens:
            for oid in token.features[feature]:
                if test_to_sound[oid] not in token.features['sound']:
                    discrepancies.append((token, oid))
        assert not discrepancies
