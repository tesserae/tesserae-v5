"""Match Greek units to Latin units"""
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix

from tesserae.data import load_greek_to_latin
from tesserae.db.entities import Feature, Match
from tesserae.matchers.sparse_encoding import \
    _get_units, _inverse_averaged_freq_getter, _lookup_wrapper, \
    gen_hits2positions, _get_distance_by_span, _get_distance_by_least_frequency
from tesserae.utils.calculations import \
    get_corpus_frequencies, get_feature_counts_by_text, \
    get_inverse_text_frequencies
from tesserae.utils.retrieve import TagHelper
from tesserae.utils.stopwords import get_feature_indices


class GreekToLatinSearch:
    matcher_type = 'greek_to_latin'

    def __init__(self, connection):
        self.connection = connection
        self.greek_to_latin = load_greek_to_latin()

    def match(self,
              search,
              greek_text_options,
              latin_text_options,
              greek_stopwords,
              latin_stopwords,
              freq_basis='texts',
              max_distance=10,
              distance_basis='frequency',
              min_score=6):
        """Find matches between a Greek text and a Latin text

        Texts will contain lines or phrases with matching tokens, with varying
        degrees of strength to the match.

        Parameters
        ----------
        search : tesserae.db.entities.Search
            The search job associated with this matching job.
        greek_text_options : tesserae.matchers.text_options.TextOptions
            The Greek text to compare against, specifying by which units.
        latin_text_options : tesserae.matchers.text_options.TextOptions
            The Latin text to compare against, specifying by which units.
        greek_stopwords : list of str
            A list of Greek words to ignore in the Greek text.
        latin_stopwords : list of str
            A list of Latin words to ignore in the Latin text.
        freq_basis : {'texts','corpus'}
            Take frequencies from the texts being matched or from the entire
            corpus.
        max_distance : int
            The maximum inter-word distance to use in a match.
        distance_basis : {'frequency', 'span'}
            The methods used to compute distance.
            - 'frequency': the distance between the two least frequent words
            - 'span': the greatest distance between any two matching words
        min_score : float
            The minimum score a match must have in order to be returned as a
            result

        Raises
        ------
        ValueError
            Raised when a parameter was poorly specified
        """
        if freq_basis != 'texts':
            raise ValueError(f'Unsupported freq_basis: {freq_basis}')
        assert greek_text_options.text.language == 'greek'
        assert latin_text_options.text.language == 'latin'
        greek_stoplist_set = set(
            get_feature_indices(self.connection, 'greek', 'lemmata',
                                greek_stopwords))
        latin_stoplist_set = set(
            get_feature_indices(self.connection, 'latin', 'lemmata',
                                latin_stopwords))
        greek_features = self.connection.find(Feature.collection,
                                              language='greek',
                                              feature='lemmata')
        greek_features.sort(key=lambda x: x.index)
        latin_features = self.connection.find(Feature.collection,
                                              language='latin',
                                              feature='lemmata')
        latin_features.sort(key=lambda x: x.index)
        greekind_to_other_greekinds = _build_greekind_to_other_greekinds(
            self.connection, self.greek_to_latin)
        valid_latin_tokens_to_indices = {
            f.token: f.index
            for f in latin_features if f.index not in latin_stoplist_set
        }

        greek_units = _get_units(self.connection, greek_text_options,
                                 'lemmata')
        latin_units = _get_units(self.connection, latin_text_options,
                                 'lemmata')

        tag_helper = TagHelper(
            self.connection,
            [greek_text_options.text, latin_text_options.text])

        greek_inv_frequencies_getter = _get_inv_greek_to_latin_freq_getter(
            self.connection, freq_basis, greek_text_options,
            greek_units,
            greekind_to_other_greekinds)
        latin_inv_frequencies_getter = _get_inv_lemmata_freq_getter(
            self.connection, freq_basis, latin_text_options, latin_units)

        search_id = search.id

        match_ents = []
        numerator_sparse_rows = []
        numerator_sparse_cols = []
        numerator_sparse_data = []
        denominators = []
        for greek_ind, latin_ind, positions in _gen_greek_to_latin_matches(
                search, self.connection, greek_units, greek_features,
                greek_stoplist_set, self.greek_to_latin,
                valid_latin_tokens_to_indices, latin_units, latin_features,
                latin_stoplist_set):
            greek_unit = greek_units[greek_ind]
            latin_unit = latin_units[latin_ind]
            greek_forms = np.array(greek_unit['forms'])
            latin_forms = np.array(latin_unit['forms'])
            greek_positions = positions[:, 0]
            latin_positions = positions[:, 1]
            if distance_basis == 'span':
                greek_distance = _get_distance_by_span(greek_positions,
                                                       greek_forms)
                latin_distance = _get_distance_by_span(latin_positions,
                                                       latin_forms)
            else:
                greek_distance = _get_distance_by_least_frequency(
                    greek_inv_frequencies_getter, greek_positions, greek_forms)
                latin_distance = _get_distance_by_least_frequency(
                    latin_inv_frequencies_getter, latin_positions, latin_forms)
            if greek_distance <= 0 or latin_distance <= 0:
                continue
            distance = greek_distance + latin_distance
            if distance <= max_distance:
                matched_greek_to_latin_features = \
                    _get_matched_greek_to_latin_features(
                        greek_unit['features'], greek_positions,
                        greek_features, self.greek_to_latin,
                        valid_latin_tokens_to_indices
                    )
                matched_latin_features = [
                    latin_unit['features'][latin_pos]
                    for latin_pos in latin_positions
                ]
                match_features = _get_match_features(
                    matched_greek_to_latin_features, matched_latin_features,
                    latin_stoplist_set)
                if match_features:
                    match_inv_frequencies = [
                        greek_inv_frequencies_getter(greek_forms[pos])
                        for pos in set(greek_positions)
                    ]
                    match_inv_frequencies.extend([
                        latin_inv_frequencies_getter(latin_forms[pos])
                        for pos in set(latin_positions)
                    ])
                    numerator_sparse_rows.extend([len(match_ents)] *
                                                 len(match_inv_frequencies))
                    numerator_sparse_cols.extend(
                        [i for i in range(len(match_inv_frequencies))])
                    numerator_sparse_data.extend(match_inv_frequencies)
                    denominators.append(distance)
                    match_ents.append(
                        Match(search_id=search_id,
                              source_unit=greek_unit['_id'],
                              target_unit=latin_unit['_id'],
                              source_tag=tag_helper.get_display_tag(
                                  greek_unit['text'], greek_unit['tags']),
                              target_tag=tag_helper.get_display_tag(
                                  latin_unit['text'], latin_unit['tags']),
                              matched_features=[
                                  latin_features[int(mf)].token
                                  for mf in match_features
                              ],
                              source_snippet=greek_unit['snippet'],
                              target_snippet=latin_unit['snippet'],
                              highlight=[(int(greek_pos), int(latin_pos))
                                         for greek_pos, latin_pos in zip(
                                             greek_positions, latin_positions)
                                         ]))
        if match_ents:
            numerators = csr_matrix((numerator_sparse_data,
                                     (numerator_sparse_rows,
                                      numerator_sparse_cols))).sum(axis=-1).A1
            scores = np.log(numerators) - np.log(denominators)
            for match, score in zip(match_ents, scores):
                match.score = score
        return match_ents


def _reverse_mapping(a2bs):
    result = defaultdict(set)
    for a, bs in a2bs.items():
        for b in bs:
            result[b].add(a)
    return result


def _build_greekind_to_other_greekinds(conn, greek_to_latin):
    greek_token_to_form = {
        f.token: f
        for f in conn.find(
            Feature.collection, language='greek', feature='form')
    }
    latin_to_greek = _reverse_mapping(greek_to_latin)
    result = defaultdict(set)
    for greek_token, latin_translations in greek_to_latin.items():
        if greek_token in greek_token_to_form:
            other_greekinds = set()
            for latin_token in latin_translations:
                for other_greek_token in latin_to_greek[latin_token]:
                    if other_greek_token in greek_token_to_form:
                        other_greekinds.add(
                            greek_token_to_form[other_greek_token].index)
            result[greek_token_to_form[greek_token].index] = other_greekinds
    return result


def _get_inv_lemmata_freq_getter(conn, freq_basis, text_options, latin_units):
    if freq_basis != 'texts':
        return _inverse_averaged_freq_getter(
            get_corpus_frequencies(conn, 'lemmata',
                                   text_options.text.language),
            latin_units)
    return _lookup_wrapper(
        get_inverse_text_frequencies(conn, 'lemmata', text_options.text.id))


def _get_greek_to_latin_inv_freqs_by_text(conn, text_options, text_length,
                                          greekind_to_other_greekinds):
    greek_lemma_counts = get_feature_counts_by_text(conn, 'lemmata',
                                                    text_options.text)
    result = {}
    for greek_form_ind, greek_counts in greek_lemma_counts.items():
        already_seen = set([greek_form_ind])
        value = greek_counts
        for other_greekind in greekind_to_other_greekinds[greek_form_ind]:
            if other_greekind in already_seen:
                continue
            value += greek_lemma_counts[other_greekind]
            already_seen.add(other_greekind)
        if value > 0:
            result[greek_form_ind] = float(text_length) / float(value)
    return result


def _get_inv_greek_to_latin_freq_getter(conn, freq_basis, text_options,
                                        greek_units,
                                        greekind_to_other_greekinds):
    if freq_basis != 'texts':
        return _inverse_averaged_freq_getter(
            get_corpus_frequencies(conn, 'lemmata',
                                   text_options.text.language),
            greek_units)
    # otherwise, handle text case
    text_length = sum(len(u['forms']) for u in greek_units)
    return _lookup_wrapper(
        _get_greek_to_latin_inv_freqs_by_text(conn, text_options, text_length,
                                              greekind_to_other_greekinds))


def make_latinized_greek_matrix(greek_units, greek_features,
                                greek_stoplist_set, greek_to_latin,
                                valid_latin_tokens_to_indices,
                                latin_features_size):
    latin_feature_inds = []
    pos_inds = []
    break_inds = [0]
    for unit in greek_units:
        end_break_inds = break_inds[-1]
        cur_features = unit['features']
        for i, features in enumerate(cur_features):
            valid_greek_features = [
                f for f in features if f not in greek_stoplist_set and f >= 0
            ]
            translated_tokens = [
                greek_to_latin[greek_features[f].token]
                for f in valid_greek_features
                if greek_features[f].token in greek_to_latin
            ]
            valid_latin_features = [
                valid_latin_tokens_to_indices[latin_token]
                for translations in translated_tokens
                for latin_token in translations
                if latin_token in valid_latin_tokens_to_indices
            ]
            latin_feature_inds.extend(valid_latin_features)
            pos_inds.extend([end_break_inds + i] * len(valid_latin_features))
        break_inds.append(end_break_inds + len(cur_features))
    return (csr_matrix(
        (np.ones(len(pos_inds), dtype=np.bool),
         (pos_inds, latin_feature_inds)),
        shape=(break_inds[-1], latin_features_size)), np.array(break_inds))


def _gen_greek_to_latin_matches(search, conn, greek_units, greek_features,
                                greek_stoplist_set, greek_to_latin,
                                valid_latin_tokens_to_indices, latin_units,
                                latin_features, latin_stoplist_set):
    latinized_greek_matrix, greek_break_inds = make_latinized_greek_matrix(
        greek_units, greek_features, greek_stoplist_set, greek_to_latin,
        valid_latin_tokens_to_indices, len(latin_features))

    for hits2positions in gen_hits2positions(search, conn,
                                             latinized_greek_matrix,
                                             greek_break_inds, latin_units,
                                             latin_stoplist_set,
                                             len(latin_features)):
        overhits2positions = {
            k: np.array(v)
            for k, v in hits2positions.items() if len(v) >= 2
        }
        for (t_ind, s_ind), positions in overhits2positions.items():
            yield (t_ind, s_ind, positions)


def _get_matched_greek_to_latin_features(greek_unit_features, greek_positions,
                                         greek_features, greek_to_latin,
                                         valid_latin_tokens_to_indices):
    result = []
    for greek_pos in greek_positions:
        greek_features_by_pos = greek_unit_features[greek_pos]
        cur_pos_latin_features = []
        for greek_feature_index in greek_features_by_pos:
            greek_token = greek_features[greek_feature_index].token
            if greek_token in greek_to_latin:
                translations = greek_to_latin[greek_token]
                for latin_token in translations:
                    if latin_token in valid_latin_tokens_to_indices:
                        cur_pos_latin_features.append(
                            valid_latin_tokens_to_indices[latin_token])
        result.append(cur_pos_latin_features)
    return result


def _get_match_features(matched_greek_to_latin_features,
                        matched_latin_features, latin_stoplist_set):
    result = set()
    for features_by_pos1, features_by_pos2 in \
            zip(matched_greek_to_latin_features, matched_latin_features):
        features_in_both = set(features_by_pos1).intersection(
            set(features_by_pos2))
        for feature in features_in_both:
            result.add(feature)
    return result - latin_stoplist_set
