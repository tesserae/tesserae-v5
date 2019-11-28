"""Unit matching by sparse matrix encoding.

Classes
-------

"""
from collections import Counter
import itertools
import multiprocessing as mp

from bson import ObjectId
import numpy as np
import pymongo
from scipy.sparse import dok_matrix

from tesserae.db.entities import Entity, Feature, Match, MatchSet, Token, Text, Unit


class SparseMatrixSearch(object):
    matcher_type = 'original'

    def __init__(self, connection):
        self.connection = connection

    def get_stoplist(self, stopwords_list, feature=None, language=None):
        """Retrieve ObjectIds for the given stopwords list

        Parameters
        ----------
        stopwords_list : list of str
            Words to consider as stopwords; these must be in normalized form

        Returns
        -------
        stoplist : list of ObjectId
            The `n` most frequent tokens in the basis texts.
        """
        pipeline = [
            {'$match': {
                'token': {'$in': stopwords_list}
            }},
            {'$project': {
                '_id': False,
                'index': True
            }}
        ]

        if language is not None:
            pipeline[0]['$match']['language'] = language

        if feature is not None:
            pipeline[0]['$match']['feature'] = feature

        stoplist = self.connection.aggregate(Feature.collection, pipeline, encode=False)
        return np.array([s['index'] for s in stoplist], dtype=np.uint32)

    def create_stoplist(self, n, feature, language, basis='corpus'):
        """Compute a stoplist of `n` tokens.

        Parameters
        ----------
        n : int
            The number of tokens to include in the stoplist.
        basis : list of tesserae.db.entities.Text, optional
            The texts to use as the frequency basis. If None, use frequencies
            across the entire corpus.

        Returns
        -------
        stoplist : list of ObjectId
            The `n` most frequent tokens in the basis texts.
        """
        pipeline = [
            {'$match': {'feature': feature, 'language': language}},
        ]

        if basis == 'corpus':
            pipeline.append(
                {'$project': {
                    '_id': False,
                    'index': True,
                    'token': True,
                    'frequency': {
                        '$reduce': {
                            'input': {'$objectToArray': '$frequencies'},
                            'initialValue': 0,
                            'in': {'$sum': ['$$value', '$$this.v']}
                        }
                    }
                }})
        else:
            basis = [t.id if isinstance(t, Entity) else t for t in basis]
            pipeline.extend([
                {'$project': {
                    '_id': False,
                    'index': True,
                    'token': True,
                    'frequency': {
                        '$sum': ['$frequencies.' + str(t_id) for t_id in basis]}
                }}
            ])

        pipeline.extend([
            {'$sort': {'frequency': -1}},
            {'$limit': n},
            {'$project': {'token': True, 'index': True, 'frequency': True}}
        ])

        stoplist = self.connection.aggregate(Feature.collection, pipeline, encode=False)
        stoplist = list(stoplist)
        print([(s['token'], s['frequency']) for s in stoplist])
        return np.array([s['index'] for s in stoplist], dtype=np.uint32)

    def match(self, texts, unit_type, feature, stopwords=10,
              stopword_basis='corpus', score_basis='word',
              frequency_basis='texts', max_distance=10,
              distance_metric='frequency', min_score=6):
        """Find matches between one or more texts.

        Texts will contain lines or phrases with matching tokens, with varying
        degrees of strength to the match. If one text is provided, each unit in
        the text will be matched with every subsequent unit.

        Parameters
        ----------
        texts : list of tesserae.db.Text
            The texts to match. Texts are matched in
        unit_type : {'line','phrase'}
            The type of unit to match on.
        feature : {'form','lemmata','semantic','lemmata + semantic','sound'}
            The token feature to match on.
        stopwords : int or list of str
            The number of stopwords to use, to be retrieved from the database,
            or else a list of words to use as stopwords.
        stopword_basis : {'corpus','texts'} or slice or tesserae.db.Text
            Which frequencies to use when calculating the stoplist.
            - 'corpus': use the combined frequencies of the entire corpus
            - 'texts': use the combined frequencies of all texts in the match
            - slice: use the texts returned from `texts` by the slice
            - Text: use a single text
        score_basis : {'word','stem'}
            Whether to score based on the words (normalized text) or stems
            (lemmata).
        frequency_basis : {'texts','corpus'}
            Take frequencies from the texts being matched or from the entire
            corpus.
        max_distance : float
            The maximum inter-word distance to use in a match.
        distance_metric : {'frequency', 'span'}
            The methods used to compute distance.
            - 'frequency': the distance between the two least frequent words
            - 'span': the greatest distance between any two matching words

        Raises
        ------
        ValueError
            Raised when a parameter was poorly specified
        """
        if isinstance(stopwords, int):
            stopword_basis = stopword_basis if stopword_basis != 'texts' \
                    else texts
            stoplist = self.create_stoplist(
                stopwords,
                'form' if feature == 'form' else 'lemmata',
                texts[0].language,
                basis=stopword_basis)
        else:
            stoplist = self.get_stoplist(stopwords,
                    'form' if feature == 'form' else 'lemmata',
                    texts[0].language)

        match_matrices = []

        feature_types = self.connection.find(Feature.collection,
                feature=feature)
        feature_count = len(feature_types)
        if feature_count <= 0:
            raise ValueError(
                f'Feature type "{feature}" was not found in the database.')

        unit_matrices = []
        unit_lists = []

        for t in texts:
            units = [u for u in self.connection.aggregate(
                Unit.collection,
                [
                    {'$match': {'text': t.id, 'unit_type': unit_type}},
                    {'$project': {
                        '_id': True,
                        'index': True,
                        'tags': True,
                        'forms': {
                            # https://docs.mongodb.com/manual/reference/operator/aggregation/reduce/
                            '$reduce': {
                                'input': '$tokens.features.form',
                                'initialValue': [],
                                'in': { '$concatArrays': ['$$value', '$$this'] }
                            }
                        },
                        'features': '$tokens.features.'+feature,
                    }}
                ],
                encode=False
            )]

            unit_indices = []
            feature_indices = []

            for unit in units:
                if 'features' in unit:
                    all_features = [
                        f_index
                        for cur_feature in unit['features']
                        for f_index in cur_feature
                    ]
                    unit_indices.extend([unit['index']
                        for _ in range(len(all_features))])
                    feature_indices.extend(all_features)

            # print(feature_indices)
            feature_matrix = dok_matrix((len(units), feature_count))
            feature_matrix[(np.asarray(unit_indices), np.asarray(feature_indices))] = 1

            del unit_indices, feature_indices

            feature_matrix = feature_matrix.tolil()
            feature_matrix[:, np.asarray(stoplist)] = 0
            feature_matrix = feature_matrix.tocsr()
            feature_matrix.eliminate_zeros()
            unit_matrices.append(feature_matrix)
            unit_lists.append(units)


        matches = unit_matrices[0].dot(unit_matrices[1].T)
        matches[matches == 1] = 0
        matches.eliminate_zeros()

        stoplist_set = set(stoplist)
        features = sorted(self.connection.find('features', language=texts[0].language, feature=feature), key=lambda x: x.index)

        if frequency_basis != 'texts':
            match_ents = _score_by_corpus_frequencies(self.connection, feature,
                    texts, matches, unit_lists, features, stoplist,
                    distance_metric, max_distance)
        else:
            match_ents = _score_by_text_frequencies(self.connection, feature,
                    texts, matches, unit_lists, features, stoplist,
                    distance_metric, max_distance)

        match_set = MatchSet(texts=texts,
            unit_types=[unit_type for _ in range(len(texts))],
            parameters={
                'method': {
                    'name': self.matcher_type,
                    'feature': feature,
                    'stopwords': [
                        self.connection.find(
                            Feature.collection, index=int(index),
                            language=texts[0].language, feature=feature)[0].token
                        for index in stoplist],
                    'freq_basis': frequency_basis,
                    'max_distance': max_distance,
                    'distance_basis': distance_metric
                }
            },
            matches=match_ents
        )

        return match_ents, match_set


def get_text_frequencies(connection, feature, text_id):
    """Get frequency data (calculated by the given feature) for words in a
    particular text.

    This method assumes that for a given word type, the feature types
    extracted from any one instance of the word type will be the same as
    all other instances of the same word type.  Thus, further work would be
    necessary, for example, if features could be extracted based on word
    position.

    Parameters
    ----------
    connection : tesserae.db.mongodb.TessMongoConnection
    feature : str
        Feature category to be used in calculating frequencies
    text_id : bson.objectid.ObjectId
        ObjectId of the text whose feature frequencies are to be computed

    Returns
    -------
    dict [int, float]
        the key should be a feature index of type form; the associated
        value is the average proportion of words in the text sharing at
        least one same feature type with the key word
    """
    tindex2mtindex = {}
    findex2mfindex = {}
    word_counts = Counter()
    word_feature_pairs = set()
    text_token_count = 0
    unit_proj = {
        '_id': False,
        'tokens.features.form': True
    }
    if feature != 'form':
        unit_proj['tokens.features.'+feature] = True
    db_cursor = connection.connection[Unit.collection].find(
        {'text': text_id, 'unit_type': 'line'},
        unit_proj
    )
    for unit in db_cursor:
        text_token_count += len(unit['tokens'])
        for token in unit['tokens']:
            cur_features = token['features']
            # use the form index as an identifier for this token's word
            # type
            cur_tindex = cur_features['form'][0]
            if cur_tindex not in tindex2mtindex:
                tindex2mtindex[cur_tindex] = len(tindex2mtindex)
            mtindex = tindex2mtindex[cur_tindex]
            # we want to count word types by matrix indices for faster
            # lookup when we get to the stage of counting up word type
            # occurrences
            word_counts[mtindex] += 1
            for cur_findex in cur_features[feature]:
                if cur_findex not in findex2mfindex:
                    findex2mfindex[cur_findex] = len(findex2mfindex)
                mfindex = findex2mfindex[cur_findex]
                # record when a word type is associated with a feature type
                word_feature_pairs.add((mtindex, mfindex))
    word_feature_matrix = dok_matrix(
        (len(tindex2mtindex), len(findex2mfindex)),
        dtype=np.bool
    )
    for mtindex, mfindex in word_feature_pairs:
        word_feature_matrix[mtindex, mfindex] = True
    word_feature_matrix.tocsr()
    # if matching_words_matrix[i, j] == True, then the word represented by
    # position i shared at least one feature type with the word represtened
    # by position j
    matching_words_matrix = word_feature_matrix.dot(
        word_feature_matrix.transpose())

    mtindex2tindex = {
        mtindex: tindex for tindex, mtindex in tindex2mtindex.items()}
    freqs = {}
    for i in range(matching_words_matrix.shape[0]):
        # since only matching tokens remain, the column indices indicate
        # which tokens match the token represented by row i; we need to
        # count up how many times each word appeared
        word_count_sum = sum([word_counts[j]
            for j in matching_words_matrix[i].nonzero()[1]])
        freqs[mtindex2tindex[i]] = word_count_sum / text_token_count
    return freqs


def get_corpus_frequencies(connection, feature, language):
    """Get frequency data for a given feature across a particular corpus

    ``feature`` refers to the kind of feature(s) extracted from the text
    (e.g., lemmata).  "feature type" refers to a group of abstract entities
    that belong to a feature (e.g., "ora" is a feature type of lemmata).
    "feature instance" refers to a particular occurrence of a feature type
    (e.g., the last word of the first line of Aeneid 1 could be derived
    from an instance of "ora"; it is also possible to have been derived
    from an instance of "os" (though Latinists typically reject this
    option)).

    This method finds the frequency of feature types by counting feature
    instances by their type and dividing each count by the total number of
    feature instances found.  Each feature type has an index associated
    with it, which should be used to look up that feature type's frequency
    in the corpus.

    Parameters
    ----------
    connection : tesserae.db.mongodb.TessMongoConnection
    feature : str
        Feature category to be used in calculating frequencies
    language : str
        Language to which the features of interest belong

    Returns
    -------
    np.array
    """
    pipeline = [
        # Get all database documents of the specified feature and language
        # (from the "features" collection, as we later find out).
        {'$match': {'feature': feature, 'language': language}},
        # Transform each document.
        {'$project': {
            # Don't keep their database identifiers,
            '_id': False,
            # but keep their indices.
            'index': True,
            # Also show their frequency,
            'frequency': {
                # which is calculated by summing over all the count values
                # found in the sub-document obtained by checking the
                # "frequencies" key in the original document.
                '$reduce': {
                    'input': {'$objectToArray': '$frequencies'},
                    'initialValue': 0,
                    'in': {'$sum': ['$$value', '$$this.v']}
                }
            }
        }},
        # Finally, sort those transformed documents by their index.
        {'$sort': {'index': 1}}
    ]

    freqs = connection.aggregate(
            Feature.collection, pipeline, encode=False)
    freqs = list(freqs)
    freqs = np.array([freq['frequency'] for freq in freqs])
    return freqs / sum(freqs)


def _score_by_corpus_frequencies(connection, feature, texts, matches,
        unit_lists, features, stoplist, distance_metric, max_distance):
    if texts[0].language != texts[1].language:
        source_frequencies_getter = _averaged_freq_getter(
            get_corpus_frequencies(connection, feature, texts[0].language),
            unit_lists[0])
        target_frequencies_getter = _averaged_freq_getter(
            get_corpus_frequencies(connection, feature, texts[1].language),
            unit_lists[1])
    else:
        source_frequencies_getter = _averaged_freq_getter(
            get_corpus_frequencies(connection, feature, texts[0].language),
            itertools.chain.from_iterable([unit_lists[0], unit_lists[1]]))
        target_frequencies_getter = source_frequencies_getter
    return _score(matches, unit_lists, features, stoplist, distance_metric,
            max_distance, source_frequencies_getter, target_frequencies_getter)


def _score_by_text_frequencies(connection, feature, texts, matches, unit_lists,
        features, stoplist, distance_metric, max_distance):
    source_frequencies_getter = _lookup_wrapper(get_text_frequencies(
            connection, feature, texts[0].id))
    target_frequencies_getter = _lookup_wrapper(get_text_frequencies(
            connection, feature, texts[1].id))
    return _score(matches, unit_lists, features, stoplist, distance_metric,
            max_distance, source_frequencies_getter, target_frequencies_getter)



def _get_distance_by_least_frequency(get_freq, positions, forms):
    """Obtains the distance by least frequency for a unit

    Contrary to the v3 help documentation on --dist in read_table.pl, v3
    behavior is that distance is inclusive of both matched words.  Thus,
    adjacent words have a distance of 2, an intervening word increases the
    distance to 3, and so forth.  This function behaves as v3 behaves instead
    of how it prescribes.

    Parameters
    ----------
    get_freq : (int) -> float
        a function that takes a word form index as input and returns its
        frequency as output
    positions : list of int
        token positions in the unit where matches were found
    form_indices : list of int
        the token forms of the unit
    """
    freqs = [get_freq(forms[i]) for i in positions]
    freq_sort = np.argsort(freqs)
    idx = np.array([positions[i] for i in freq_sort])
    if idx.shape[0] >= 2:
        not_first_pos = [s for s in idx if s != idx[0]]
        if not_first_pos:
            end = not_first_pos[0]
            return np.abs(end - idx[0]) + 1
    return 0


def _lookup_wrapper(d):
    """Useful for making dictionaries act like functions"""
    def _inner(key):
        return d[key]
    return _inner


def _averaged_freq_getter(d, units_iter):
    cache = {}
    for unit in units_iter:
        for form, feats in zip(unit['forms'], unit['features']):
            if form in cache:
                continue
            cache[form] = np.mean([d[f] for f in feats])
    def _inner(key):
        return cache[key]
    return _inner


def _score(matches, unit_lists, features, stoplist, distance_metric,
        max_distance, source_frequencies_getter, target_frequencies_getter):
    match_ents = []
    for i in range(matches.shape[0]):
        source_unit = unit_lists[0][i]
        target_units = [unit_lists[1][j] for j in matches[i].nonzero()[1]]
        # number of source positions x number of features
        feature_source_matrix = dok_matrix(
                (len(features), len(source_unit['forms'])), dtype=np.bool)
        for pos, feats in enumerate(source_unit['features']):
            feature_source_matrix[feats, pos] = True
        feature_source_matrix = feature_source_matrix.tolil()
        feature_source_matrix[np.asarray(stoplist), :] = False
        feature_source_matrix = feature_source_matrix.tocsr()
        feature_source_matrix.eliminate_zeros()
        # number of features x number of target positions
        target_feature_matrix = dok_matrix(
                (
                    sum([len(feats)
                        for target_unit in target_units
                        for feats in target_unit['features']]),
                    len(features)
                ),
                dtype=np.bool)
        unitbreak_indices = [0]
        for target_ind, target_unit in enumerate(target_units):
            for pos, feats in enumerate(target_unit['features']):
                target_feature_matrix[pos+unitbreak_indices[-1], feats] = True
            unitbreak_indices.append(
                    unitbreak_indices[-1] + len(target_unit['features']))
        target_feature_matrix = target_feature_matrix.tolil()
        target_feature_matrix[:, np.asarray(stoplist)] = False
        target_feature_matrix = target_feature_matrix.tocsr()
        target_feature_matrix.eliminate_zeros()
        pos_matched = target_feature_matrix.dot(feature_source_matrix)
        for target_ind, target_unit in enumerate(target_units):
            start_row = unitbreak_indices[target_ind]
            limit_row = unitbreak_indices[target_ind+1]
            # rows are target positions of matching tokens
            # cols are source positions of matching tokens
            rows, cols = pos_matched[start_row:limit_row].nonzero()
            target_forms = target_unit['forms']
            source_forms = source_unit['forms']
            if distance_metric == 'span':
                # adjacent matched words have a distance of 2, etc.
                target_distance = np.abs(np.max(rows) - np.min(rows)) + 1
                source_distance = np.abs(np.max(cols) - np.min(cols)) + 1
            else:
                target_distance = _get_distance_by_least_frequency(
                        target_frequencies_getter, rows,
                        target_forms)
                source_distance = _get_distance_by_least_frequency(
                        source_frequencies_getter, cols,
                        source_forms)
            distance = source_distance + target_distance
            if distance < max_distance:
                match_frequencies = [target_frequencies_getter(target_forms[pos])
                    for pos in rows]
                match_frequencies.extend(
                    [source_frequencies_getter(source_forms[pos])
                    for pos in cols])
                score = np.log((np.sum(np.power(match_frequencies, -1))) / distance)
                target_features = target_unit['features']
                source_features = source_unit['features']
                match_features = set(itertools.chain.from_iterable([
                        set(target_features[row]).intersection(
                            set(source_features[col]))
                        for row, col in zip(rows, cols)]))
                match_ents.append(Match(
                    units=[source_unit['_id'], target_unit['_id']],
                    tokens=[features[int(mf)] for mf in match_features],
                    score=score
                ))
    return match_ents
