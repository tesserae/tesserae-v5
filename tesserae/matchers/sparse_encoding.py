"""Unit matching by sparse matrix encoding.

Classes
-------

"""
from collections import Counter
import itertools
import multiprocessing as mp
import time

from bson import ObjectId
import numba
import numpy as np
import pymongo
from scipy.sparse import csr_matrix, dok_matrix

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

        features = sorted(
                self.connection.find(
                    Feature.collection, language=texts[0].language,
                    feature=feature),
                key=lambda x: x.index)
        if len(features) <= 0:
            raise ValueError(
                f'Feature type "{feature}" for language "{texts[0].language}" '
                f'was not found in the database.')

        target_units = _get_units(self.connection, [texts[1]], unit_type,
                feature)
        source_units = _get_units(self.connection, [texts[0]], unit_type,
                feature)

        if frequency_basis != 'texts':
            match_ents = _score_by_corpus_frequencies(self.connection, feature,
                    texts, target_units, source_units, features, stoplist,
                    distance_metric, max_distance)
        else:
            match_ents = _score_by_text_frequencies(self.connection, feature,
                    texts, target_units, source_units, features, stoplist,
                    distance_metric, max_distance)

        stopword_tokens = [s.token
                for s in self.connection.find(
                    Feature.collection, index=[int(i) for i in stoplist],
                    language=texts[0].language, feature=feature)]
        match_set = MatchSet(texts=texts,
            unit_types=[unit_type for _ in range(len(texts))],
            parameters={
                'method': {
                    'name': self.matcher_type,
                    'feature': feature,
                    'stopwords': stopword_tokens,
                    'freq_basis': frequency_basis,
                    'max_distance': max_distance,
                    'distance_basis': distance_metric
                }
            },
            matches=match_ents
        )

        return match_ents, match_set


def _get_units(connection, texts, unit_type, feature):
    units = []
    for t in texts:
        units.extend([u for u in connection.aggregate(
            Unit.collection,
            [
                {'$match': {'text': t.id, 'unit_type': unit_type}},
                {'$project': {
                    '_id': True,
                    'index': True,
                    'tags': True,
                    'forms': {
                        # flatten list of lists of ints into list of ints
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
        )])
    return units


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
    csr_rows = []
    csr_cols = []
    for mtindex, mfindex in word_feature_pairs:
        csr_rows.append(mtindex)
        csr_cols.append(mfindex)
    word_feature_matrix = csr_matrix(
        (np.ones(len(csr_rows), dtype=np.bool), (np.array(csr_rows),
            np.array(csr_cols))),
        shape=(len(tindex2mtindex), len(findex2mfindex))
    )
    # if matching_words_matrix[i, j] == True, then the word represented by
    # position i shared at least one feature type with the word represented
    # by position j
    matching_words_matrix = word_feature_matrix.dot(
        word_feature_matrix.transpose())

    mtindex2tindex = {
        mtindex: tindex for tindex, mtindex in tindex2mtindex.items()}
    freqs = {}
    coo = matching_words_matrix.tocoo()
    for i, j in zip(coo.row, coo.col):
        # since only matching tokens remain, the column indices indicate
        # which tokens match the token represented by row i; we need to
        # count up how many times each word appeared
        cur_token = mtindex2tindex[i]
        if cur_token not in freqs:
            freqs[cur_token] = word_counts[j]
        else:
            freqs[cur_token] += word_counts[j]
    for tok_ind in freqs:
        freqs[tok_ind] = freqs[tok_ind] / text_token_count
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


def _score_by_corpus_frequencies(connection, feature, texts, target_units,
        source_units,
        features, stoplist, distance_metric, max_distance):
    if texts[0].language != texts[1].language:
        source_frequencies_getter = _averaged_freq_getter(
            get_corpus_frequencies(connection, feature, texts[0].language),
            source_units)
        target_frequencies_getter = _averaged_freq_getter(
            get_corpus_frequencies(connection, feature, texts[1].language),
            target_units)
    else:
        source_frequencies_getter = _averaged_freq_getter(
            get_corpus_frequencies(connection, feature, texts[0].language),
            itertools.chain.from_iterable([source_units, target_units]))
        target_frequencies_getter = source_frequencies_getter
    return _score(target_units, source_units, features, stoplist,
            distance_metric,
            max_distance, source_frequencies_getter, target_frequencies_getter)


def _score_by_text_frequencies(connection, feature, texts, target_units,
        source_units,
        features, stoplist, distance_metric, max_distance):
    source_frequencies_getter = _lookup_wrapper(get_text_frequencies(
            connection, feature, texts[0].id))
    target_frequencies_getter = _lookup_wrapper(get_text_frequencies(
            connection, feature, texts[1].id))
    return _score(target_units, source_units, features, stoplist,
            distance_metric,
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


def _get_distance_by_span(matched_positions):
    """Calculate distance between two matching words

    Parameters
    ----------
    matched_positions : list of int
        the positions at which matched words were found in a unit
    """
    start_pos = np.min(matched_positions)
    end_pos = np.max(matched_positions)
    if start_pos != end_pos:
        np.abs(end_pos - start_pos) + 1
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


def _extract_features_and_positions(units, stoplist_set):
    """Grab feature and token information from units

    All features for every token within each of the units will be accounted for
    exactly once.

    Parameters
    ----------
    units
        ``units`` should be either ``source_units`` or ``target_units`` from
        ``_gen_matches(...)``
    stoplist_set : set of int
        feature indices which should not be recorded

    Returns
    -------
    feature_inds : 1d np.array of int
    pos_inds : 1d np.array of int
        ``feature_inds[i]`` should be equal to a feature index present at the
        position equal to ``pos_inds[i]``
    break_inds : 1d np.array of int
        for the slice ``break_inds[i]:break_inds[i+1]``, those are the position
        indices that belong to ``units[i]``

    Example
    -------
    >>> unit_0 = {... 'forms': [1], 'features': [[10, 20]]}
    >>> unit_1 = {... 'forms': [5, 17], 'features': [[7], [21, 30]]}
    >>> units = [unit_0, unit_1]
    >>> feature_inds, pos_inds, break_inds = \
    >>> ... _extract_features_and_positions(units, {7})
    >>> feaure_inds == np.array([10, 20, 21, 30])
    >>> pos_inds == np.array([0, 0, 2, 2])
    >>> break_inds == np.array([0, 1, 3])

    """
    feature_inds = []
    pos_inds = []
    break_inds = [0]
    for unit in units:
        end_break_inds = break_inds[-1]
        cur_features = unit['features']
        for i, features in enumerate(cur_features):
            valid_features = [f for f in features if f not in stoplist_set]
            feature_inds.extend(valid_features)
            pos_inds.extend([end_break_inds + i] * len(valid_features))
        break_inds.append(end_break_inds + len(cur_features))
    return np.array(feature_inds), np.array(pos_inds), np.array(break_inds)


def _construct_feature_unit_matrix(units, stoplist_set, features_size):
    """Build a matrix where rows correspond to features and columns to
    positions within units

    Parameters
    ----------
    units
        ``units`` should be either ``source_units`` or ``target_units`` from
        ``_gen_matches(...)``
    stoplist_set : set of int
        feature indices which should not be recorded
    features_size : int
        the total number of feature types for the class of features contained
        in ``units``

    Returns
    -------
    M : csr_matrix
        the result matrix; if ``M[i, j] == True``, the feature with index i
        appears at position j.
    break_inds : list of int
        see ``_extract_features_and_positions()`` for details

    """
    feature_inds, pos_inds, break_inds = _extract_features_and_positions(
            units, stoplist_set)
    return (
        csr_matrix(
            (np.ones(len(pos_inds), dtype=np.bool), (feature_inds, pos_inds)),
            shape=(features_size, break_inds[-1])),
        break_inds
    )


def _construct_unit_feature_matrix(units, stoplist_set, features_size):
    """Build a matrix where rows correspond to positions within units and
    columns to features

    Parameters
    ----------
    units
        ``units`` should be either ``source_units`` or ``target_units`` from
        ``_gen_matches(...)``
    stoplist_set : set of int
        feature indices which should not be recorded
    features_size : int
        the total number of feature types for the class of features contained
        in ``units``

    Returns
    -------
    M : csr_matrix
        the result matrix; if ``M[i, j] == True``, the position i has feature j
        appear
    break_inds : list of int
        see ``_extract_features_and_positions()`` for details

    """
    feature_inds, pos_inds, break_inds = _extract_features_and_positions(
            units, stoplist_set)
    return (
        csr_matrix(
            (np.ones(len(pos_inds), dtype=np.bool), (pos_inds, feature_inds)),
            shape=(break_inds[-1], features_size)),
        break_inds
    )


#@numba.njit
def _bin_hits_to_unit_indices(rows, cols, target_breaks, source_breaks):
    """Extract which units matched from the ``match_matrix``

    Parameters
    ----------
    rows : 1d np.array of ints
    cols : 1d np.array of ints
    target_breaks : 1d np.array of ints
        ``target_breaks[t]`` tells which row target unit t starts on; thus, the
        range ``target_breaks[t]:target_breaks[t+1]`` includes all the rows
        which belong to target unit t; ``len(target_breaks)`` is equal to one
        more than the total number of target units
    source_breaks : 1d np.array of ints
        like ``target_breaks``, except it keeps track of which columns belong
        to which source unit

    Returns
    -------
    hits2t_positions : dict [(int, int), 1d np.array of ints]
        the key is a tuple, where the first value refers to the index of a
        target unit and the second value refers to the index of a source unit;
        the associated 1d array tells which positions within the target unit
        had a match; the 1d array associated with the same key in
        hits2s_positions corresponds:  the xth int of the
        ``hits2t_positions[key]`` is the position in the target unit that
        matched with the position of the source unit recorded in the xth
        int of ``hits2s_positions[key]``
    hits2s_positions : dict [(int, int), 1d np.array of ints]
        the key is a tuple, where the first value refers to the index of a
        target unit and the second value refers to the index of a source unit;
        the associated 1d array tells which positions within the source unit
        had a match; the 1d array associated with the same key in
        hits2t_positions corresponds:  the xth int of the
        ``hits2s_positions[key]`` is the position in the source unit that
        matched with the position of the target unit recorded in the xth
        int of ``hits2t_positions[key]``

    Example
    -------
    >>> target_breaks = [0, 2]
    >>> source_breaks = [0, 3]
    >>> match_matrix = csr_matrix([
    >>> ... [True, False, False],
    >>> ... [False, False, True]
    >>> ... ])
    >>> coo = match_matrix.tocoo()
    >>> hits2t_positions, hits2s_positions = _bin_hits_to_unit_indices(
    >>> ... coo.rows, coo.cols, target_breaks, target_breaks)
    >>> hits2t_positions[(0, 0)] == np.array([0, 1])
    >>> hits2s_positions[(0, 0)] == np.array([0, 2])

    """
    # keep track of mapping between matrix row index and target unit index
    # in ``target_units``
    row2t_unit_ind = np.array([u_ind
            for u_ind in range(len(target_breaks) - 1)
            for _ in range(target_breaks[u_ind+1] - target_breaks[u_ind])])
    # keep track of mapping between matrix column index and source unit index
    # in ``source_units``
    col2s_unit_ind = np.array([u_ind
            for u_ind in range(len(source_breaks) - 1)
            for _ in range(source_breaks[u_ind+1] - source_breaks[u_ind])])
    hits2t_positions = {}
    hits2s_positions = {}
    tmp_stash = {}
    # TODO make faster by going through only the rows and columns
    # that will yield a match?
    for i, j in zip(rows, cols):
        # assume that ``matched_matrix[i, j] == True`` for all i and j here
        t_ind = row2t_unit_ind[i]
        s_ind = col2s_unit_ind[j]
        t_pos = i - target_breaks[t_ind]
        s_pos = j - source_breaks[s_ind]
        key = (t_ind, s_ind)
        if key in hits2t_positions:
            hits2t_positions[key] = np.append(hits2t_positions[key], [t_pos])
            hits2s_positions[key] = np.append(hits2s_positions[key], [s_pos])
        elif key in tmp_stash:
            stashed = tmp_stash[key]
            hits2t_positions[key] = np.array([stashed[0], t_pos])
            hits2s_positions[key] = np.array([stashed[1], s_pos])
        else:
            tmp_stash[key] = np.array([t_pos, s_pos])
    return hits2t_positions, hits2s_positions


def _gen_matches(target_units, source_units, stoplist, features_size):
    """Generate matching units based on unit information

    Parameters
    ----------
    source_units : list of dict
        each dictionary represents unit information from the source text
    target_units : list of dict
        each dictionary represents unit information from the target text
    stoplist : list of int
        feature indices on which matches should not be permitted
    features_size : int
        the total number of feature types for the class of features contained
        in ``units``

    Notes
    -----
    The dictionaries in the input lists must contain the following string keys
    and corresponding values:
    '_id' : bson.objectid.ObjectId
        ObjectId of the Unit entity in the database
    'index' : int
        index of the Unit entity in the database
    'tags' : list of str
        tag information for this unit
    'forms' : list of int
        the ``...['forms'][y]`` is the integer associated with the form
        Feature of the word token at position y
    'features' : list of list of int
        each position of this list corresponds to the same position in the
        'forms' list; thus, ...['features'][a] is a list of the Feature indices
        derived from ...['forms'][a].

    Yields
    ------
    target_unit : dict
    target_positions_matched : list of ints
    source_unit : dict
    source_positions_matched : list of ints

    """
    stoplist_set = set(stoplist)
    feature_source_matrix, source_breaks = _construct_feature_unit_matrix(
            source_units, stoplist_set, features_size)
    target_feature_matrix, target_breaks = _construct_unit_feature_matrix(
            target_units, stoplist_set, features_size)
    # for every position of each target unit, this matrix multiplication picks
    # up which source unit positions shared at least one common feature
    match_matrix = target_feature_matrix.dot(feature_source_matrix)
    # this data structure keeps track of which target unit position matched
    # with which source unit position
    coo = match_matrix.tocoo()
    hits2t_positions, hits2s_positions = _bin_hits_to_unit_indices(
            coo.row, coo.col, target_breaks, source_breaks)
    print(len(hits2t_positions))
    for (t_ind, s_ind), t_positions in hits2t_positions.items():
        if len(t_positions) >= 2:
            yield (target_units[t_ind], t_positions,
                source_units[s_ind], hits2s_positions[(t_ind, s_ind)])


def _score(target_units, source_units, features, stoplist, distance_metric,
        max_distance, source_frequencies_getter, target_frequencies_getter):
    match_ents = []
    features_size = len(features)
    for target_unit, t_positions, source_unit, s_positions in _gen_matches(
            target_units, source_units, stoplist, features_size):
        target_forms = target_unit['forms']
        source_forms = source_unit['forms']
        if distance_metric == 'span':
            # adjacent matched words have a distance of 2, etc.
            target_distance = _get_distance_by_span(t_positions)
            source_distance = _get_distance_by_span(s_positions)
        else:
            target_distance = _get_distance_by_least_frequency(
                    target_frequencies_getter, t_positions,
                    target_forms)
            source_distance = _get_distance_by_least_frequency(
                    source_frequencies_getter, s_positions,
                    source_forms)
        if source_distance <= 0 or target_distance <= 0:
            # less than two matching tokens in one of the units
            continue
        distance = source_distance + target_distance
        if distance < max_distance:
            match_frequencies = [target_frequencies_getter(target_forms[pos])
                for pos in t_positions]
            match_frequencies.extend(
                [source_frequencies_getter(source_forms[pos])
                for pos in s_positions])
            score = np.log((np.sum(np.power(match_frequencies, -1))) / distance)
            target_features = target_unit['features']
            source_features = source_unit['features']
            match_features = set(itertools.chain.from_iterable([
                    set(target_features[t_pos]).intersection(
                        set(source_features[s_pos]))
                    for t_pos, s_pos in zip(t_positions, s_positions)]))
            match_ents.append(Match(
                units=[source_unit['_id'], target_unit['_id']],
                tokens=[features[int(mf)] for mf in match_features],
                score=score
            ))
    return match_ents
