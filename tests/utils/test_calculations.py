import itertools
import math

from tesserae.db.entities import Feature, Text
from tesserae.utils.calculations import \
    get_corpus_frequencies, get_feature_counts_by_text, \
    get_inverse_text_frequencies


def _load_v3_mini_text_freqs_file(conn, text, v3feature):
    db_cursor = conn.connection[Feature.collection].find(
            {'feature': 'form', 'language': text.language},
            {'_id': False, 'index': True, 'token': True})
    token2index = {e['token']: e['index'] for e in db_cursor}
    # the .freq_score_* file is named the same as its corresponding
    # .tess file
    counts_path = text.path[:-4] + 'freq_score_' + v3feature
    counts = {}
    with open(counts_path, 'r', encoding='utf-8') as ifh:
        for line in ifh:
            if line.startswith('# count:'):
                total = int(line.split()[-1])
                break
        for line in ifh:
            line = line.strip()
            if line:
                word, count = line.split()
                counts[token2index[word]] = int(count)
    return total, counts


def _load_v3_mini_text_counts(conn, text):
    _, counts = _load_v3_mini_text_freqs_file(conn, text, 'word')
    return counts


def test_mini_get_feature_counts_by_text(minipop, v3checker):
    for text in minipop.find(Text.collection):
        v5_counts = get_feature_counts_by_text(minipop, 'form', text)
        v3_counts = _load_v3_mini_text_counts(minipop, text)
        in_v5 = set(v5_counts.keys())
        in_v3 = set(v3_counts.keys())
        assert len(in_v5) == len(in_v3)
        assert len(in_v5 - in_v3) == 0
        assert len(in_v3 - in_v5) == 0
        for feature_index, v5_count in v5_counts.items():
            assert v5_count == v3_counts[feature_index]


def _load_v3_mini_text_stem_freqs(conn, text):
    denom, counts = _load_v3_mini_text_freqs_file(conn, text, 'stem')
    return {token: float(count) / denom for token, count in counts.items()}


def test_mini_text_frequencies(
        minipop, mini_latin_metadata, mini_greek_metadata):
    all_text_metadata = [m for m in itertools.chain.from_iterable(
        [mini_latin_metadata, mini_greek_metadata])]
    title2id = {t.title: t.id for t in minipop.find(
        Text.collection, title=[m['title'] for m in all_text_metadata])}
    for metadata in all_text_metadata:
        v3freqs = _load_v3_mini_text_stem_freqs(
            minipop, Text.json_decode(metadata)
        )
        text_id = title2id[metadata['title']]
        v5freqs = get_inverse_text_frequencies(minipop, 'lemmata', text_id)
        for form_index, freq in v5freqs.items():
            assert form_index in v3freqs
            assert math.isclose(v3freqs[form_index], 1.0 / freq)


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
