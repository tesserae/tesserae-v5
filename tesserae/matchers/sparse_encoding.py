"""Unit matching by sparse matrix encoding.

Classes
-------

"""
import itertools

import numpy as np
from scipy.sparse import csr_matrix

from tesserae.db.entities import Feature, Match, Unit
from tesserae.utils.calculations import \
    get_corpus_frequencies, get_inverse_text_frequencies
from tesserae.utils.retrieve import TagHelper
from tesserae.utils.stopwords import create_stoplist, get_stoplist_indices


class SparseMatrixSearch(object):
    matcher_type = 'original'

    def __init__(self, connection):
        self.connection = connection

    @staticmethod
    def paramify(search_params):
        """Make JSONizable parameters for SparseMatrixSearch

        To ensure consistent storage of parameters of this search type, Search
        entities will store the search parameters returned by this method.

        Parameters
        ----------
        search_params : dict

        Returns
        -------
        dict
        """
        return {
            'source': {
                'object_id': str(search_params['source'].text.id),
                'units': search_params['source'].unit_type
            },
            'target': {
                'object_id': str(search_params['target'].text.id),
                'units': search_params['target'].unit_type
            },
            'method': {
                'name': SparseMatrixSearch.matcher_type,
                'feature': search_params['feature'],
                'stopwords': search_params['stopwords'],
                'freq_basis': search_params['freq_basis'],
                'max_distance': search_params['max_distance'],
                'distance_basis': search_params['distance_basis'],
                'min_score': search_params['min_score']
            }
        }

    @staticmethod
    def get_agg_query(source, target, method):
        """Make aggregation pipeline query parameters

        Running an aggregation pipeline with the returned dictionary should
        identify any cached results in the database for a search that used the
        specified search parameters.

        Parameters
        ----------
        source
        target
        method

        Returns
        -------
        dict
        """
        return {
            'parameters.source.object_id':
            str(source['object_id']),
            'parameters.source.units':
            source['units'],
            'parameters.target.object_id':
            str(target['object_id']),
            'parameters.target.units':
            target['units'],
            'parameters.method.name':
            method['name'],
            'parameters.method.feature':
            method['feature'],
            '$and': [{
                'parameters.method.stopwords': {
                    '$all': method['stopwords']
                }
            }, {
                'parameters.method.stopwords': {
                    '$size': len(method['stopwords'])
                }
            }],
            'parameters.method.freq_basis':
            method['freq_basis'],
            'parameters.method.max_distance':
            method['max_distance'],
            'parameters.method.distance_basis':
            method['distance_basis'],
            'parameters.method.min_score':
            method['min_score']
        }

    def match(self,
              search,
              source,
              target,
              feature,
              stopwords=10,
              stopword_basis='corpus',
              score_basis='word',
              freq_basis='texts',
              max_distance=10,
              distance_basis='frequency',
              min_score=6):
        """Find matches between one or more texts.

        Texts will contain lines or phrases with matching tokens, with varying
        degrees of strength to the match. If one text is provided, each unit in
        the text will be matched with every subsequent unit.

        Parameters
        ----------
        search : tesserae.db.entities.Search
            The search job associated with this matching job.
        source : tesserae.matchers.text_options.TextOptions
            The source text to compare against, specifying by which units.
        target : tesserae.matchers.text_options.TextOptions
            The target text to compare against, specifying by which units.
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
        freq_basis : {'texts','corpus'}
            Take frequencies from the texts being matched or from the entire
            corpus.
        max_distance : float
            The maximum inter-word distance to use in a match.
        distance_basis : {'frequency', 'span'}
            The methods used to compute distance.
            - 'frequency': the distance between the two least frequent words
            - 'span': the greatest distance between any two matching words
        min_score : float
            The minimum score a match must have in order to be included in the
            results

        Raises
        ------
        ValueError
            Raised when a parameter was poorly specified

        Returns
        -------
        list of tesserae.db.entities.Match
        """
        texts = [source.text, target.text]
        if isinstance(stopwords, int):
            stopword_basis = stopword_basis if stopword_basis != 'texts' \
                    else texts
            stoplist = create_stoplist(
                self.connection,
                stopwords,
                'form' if feature == 'form' else 'lemmata',
                source.text.language,
                basis=stopword_basis)
        else:
            stoplist = get_stoplist_indices(
                self.connection,
                stopwords,
                'form' if feature == 'form' else 'lemmata',
                source.text.language,
            )

        features = sorted(self.connection.find(Feature.collection,
                                               language=source.text.language,
                                               feature=feature),
                          key=lambda x: x.index)
        if len(features) <= 0:
            raise ValueError(f'Feature type "{feature}" for language '
                             f'"{source.text.language}" '
                             f'was not found in the database.')

        target_units = _get_units(self.connection, target, feature)
        source_units = _get_units(self.connection, source, feature)

        tag_helper = TagHelper(self.connection, texts)

        if freq_basis != 'texts':
            match_ents = _score_by_corpus_frequencies(search, self.connection,
                                                      feature, texts,
                                                      target_units,
                                                      source_units, features,
                                                      stoplist, distance_basis,
                                                      max_distance, tag_helper)
        else:
            match_ents = _score_by_text_frequencies(search, self.connection,
                                                    feature, texts,
                                                    target_units, source_units,
                                                    features, stoplist,
                                                    distance_basis,
                                                    max_distance, tag_helper)

        return [m for m in match_ents if m.score >= min_score]


def _get_units(connection, textoptions, feature):
    return [
        u for u in connection.aggregate(
            Unit.collection,
            [
                {
                    '$match': {
                        'text': textoptions.text.id,
                        'unit_type': textoptions.unit_type
                    }
                },
                {
                    '$project': {
                        '_id': True,
                        'text': True,
                        'index': True,
                        'snippet': True,
                        'tags': True,
                        'forms': {
                            # flatten list of lists of ints into list of ints
                            # https://docs.mongodb.com/manual/reference/operator/aggregation/reduce/
                            '$reduce': {
                                'input': '$tokens.features.form',
                                'initialValue': [],
                                'in': {
                                    '$concatArrays': ['$$value', '$$this']
                                }
                            }
                        },
                        'features': '$tokens.features.' + feature,
                    }
                }
            ],
            encode=False)
    ]


def _score_by_corpus_frequencies(search, connection, feature, texts,
                                 target_units, source_units, features,
                                 stoplist, distance_basis, max_distance,
                                 tag_helper):
    if texts[0].language != texts[1].language:
        source_inv_frequencies_getter = _inverse_averaged_freq_getter(
            get_corpus_frequencies(connection, feature, texts[0].language),
            source_units)
        target_inv_frequencies_getter = _inverse_averaged_freq_getter(
            get_corpus_frequencies(connection, feature, texts[1].language),
            target_units)
    else:
        source_inv_frequencies_getter = _inverse_averaged_freq_getter(
            get_corpus_frequencies(connection, feature, texts[0].language),
            itertools.chain.from_iterable([source_units, target_units]))
        target_inv_frequencies_getter = source_inv_frequencies_getter
    return _score(search, connection, target_units, source_units, features,
                  stoplist, distance_basis, max_distance,
                  source_inv_frequencies_getter, target_inv_frequencies_getter,
                  tag_helper)


def _score_by_text_frequencies(search, connection, feature, texts,
                               target_units, source_units, features, stoplist,
                               distance_basis, max_distance, tag_helper):
    source_frequencies_getter = _lookup_wrapper(
        get_inverse_text_frequencies(connection, feature, texts[0].id))
    target_frequencies_getter = _lookup_wrapper(
        get_inverse_text_frequencies(connection, feature, texts[1].id))
    return _score(search, connection, target_units, source_units, features,
                  stoplist, distance_basis, max_distance,
                  source_frequencies_getter, target_frequencies_getter,
                  tag_helper)


def _get_trivial_distance(p0, p1):
    """Calculates the distance between two positions

    Parameters
    ----------
    p0, p1 : ints
        token positions in the unit where matches were found
    """
    if p0 == p1:
        return 0
    return abs(p0 - p1) + 1


def _get_distance_by_least_frequency(get_inv_freq, positions, forms):
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
    positions : 1d np.array of ints
        token positions in the unit where matches were found
    forms : 1d np.array of ints
        the token forms of the unit
    """
    if len(set(forms[positions])) < 2:
        return 0
    if len(positions) == 2:
        return _get_trivial_distance(positions[0], positions[1])
    sorted_positions = np.array(sorted(positions))
    inv_freqs = np.array([get_inv_freq(f) for f in forms[sorted_positions]])
    # lowest inverse frequencies are the highest frequencies, so need to flip
    freq_sort = np.argsort(-inv_freqs)
    idx = sorted_positions[freq_sort]
    if idx.shape[0] >= 2:
        not_first_pos = idx[idx != idx[0]]
        if not_first_pos.shape[0] > 0:
            end = not_first_pos[0]
            return np.abs(end - idx[0]) + 1
    return 0


def _get_distance_by_span(matched_positions, forms):
    """Calculate distance between two matching words

    Parameters
    ----------
    matched_positions : 1d np.array of ints
        the positions at which matched words were found in a unit
    forms : 1d np.array of ints
        the token forms of the unit
    """
    if len(set(forms[matched_positions])) < 2:
        return 0
    if len(matched_positions) == 2:
        return _get_trivial_distance(matched_positions[0],
                                     matched_positions[1])
    start_pos = np.min(matched_positions)
    end_pos = np.max(matched_positions)
    if start_pos != end_pos:
        return np.abs(end_pos - start_pos) + 1
    return 0


def _lookup_wrapper(d):
    """Useful for making dictionaries act like functions"""
    def _inner(key):
        return d[key]

    return _inner


def _inverse_averaged_freq_getter(d, units_iter):
    cache = {}
    for unit in units_iter:
        for form, feats in zip(unit['forms'], unit['features']):
            if form in cache:
                continue
            cache[form] = 1.0 / np.mean([d[f] for f in feats])

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
            valid_features = [
                f for f in features if f not in stoplist_set and f >= 0
            ]
            if -1 in valid_features:
                print(unit)
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
    for sw in stoplist_set:
        if np.any(feature_inds == sw):
            raise Exception('Stopword in Feature x Unit Matrix!')
    return (csr_matrix(
        (np.ones(len(pos_inds), dtype=np.bool), (feature_inds, pos_inds)),
        shape=(features_size, break_inds[-1])), break_inds)


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
    for sw in stoplist_set:
        if np.any(feature_inds == sw):
            raise Exception('Stopword in Unit x Feature Matrix!')
    return (csr_matrix(
        (np.ones(len(pos_inds), dtype=np.bool), (pos_inds, feature_inds)),
        shape=(break_inds[-1], features_size)), break_inds)


def _bin_hits_to_unit_indices(rows, cols, row2t_unit_ind, target_breaks,
                              source_breaks, su_start):
    """Extract which units matched from the ``match_matrix``

    Parameters
    ----------
    rows : 1d np.array of ints
        rows is all i for which ``match_matrix[i, j] == True``; also, for all
        z, ``match_matrix[rows[z], cols[z]] == True``; all other indices should
        yield False
    cols : 1d np.array of ints
        cols is all j for which ``match_matrix[i, j] == True``; also, for all
        z, ``match_matrix[rows[z], cols[z]] == True``; all other indices should
        yield False
    row2t_unit_ind : 1d np.array of ints
        mapping between row index of matrix and unit index of target
    target_breaks : 1d np.array of ints
        ``target_breaks[t]`` tells which row target unit t starts on; thus, the
        range ``target_breaks[t]:target_breaks[t+1]`` includes all the rows
        which belong to target unit t; ``len(target_breaks)`` is equal to one
        more than the total number of target units
    source_breaks : 1d np.array of ints
        like ``target_breaks``, except it keeps track of which columns belong
        to which source unit
    su_start : int
        an offset by which to increment source indices

    Returns
    -------
    hits2positions : dict [(int, int), 2d np.array of ints]
        the key is a tuple, where the first value refers to the index of a
        target unit and the second value refers to the index of a source unit;
        the associated 2d array tells which positions within the target and
        source units had a match; in particular, each row represent matched
        positions, where the value in the first column tells the target
        position and the the value in the second column tells the source
        position; there will always be at least two rows in the 2d array

    Example
    -------
    >>> target_breaks = [0, 2]
    >>> source_breaks = [0, 3]
    >>> match_matrix = csr_matrix([
    >>> ... [True, False, False],
    >>> ... [False, False, True]
    >>> ... ])
    >>> coo = match_matrix.tocoo()
    >>> hits2positions = _bin_hits_to_unit_indices(
    >>> ... coo.rows, coo.cols, target_breaks, target_breaks)
    >>> hits2positions[(0, 0)] == np.array([[0, 0], [1, 2]])

    """
    # keep track of mapping between matrix column index and source unit index
    # in ``source_units``
    col2s_unit_ind = np.array([
        u_ind for u_ind in range(len(source_breaks) - 1)
        for _ in range(source_breaks[u_ind + 1] - source_breaks[u_ind])
    ])
    tmp = {}
    hits2positions = {}
    t_inds = row2t_unit_ind[rows]
    s_inds = col2s_unit_ind[cols]
    t_poses = rows - target_breaks[t_inds]
    s_poses = cols - source_breaks[s_inds]
    # although s_inds needs to index the source_breaks by the ordering of this
    # batch of source_units, s_inds needs to account for source_unit indices as
    # referenced from outside of this batch
    s_inds += su_start
    for t_ind, s_ind, t_pos, s_pos in zip(t_inds, s_inds, t_poses, s_poses):
        key = (t_ind, s_ind)
        if key not in tmp:
            tmp[key] = (t_pos, s_pos)
        elif key not in hits2positions:
            hits2positions[key] = [tmp[key], (t_pos, s_pos)]
        else:
            hits2positions[key].append((t_pos, s_pos))
    hits2positions = {k: np.array(v) for k, v in hits2positions.items()}
    return hits2positions


def gen_hits2positions(search, conn, target_feature_matrix, target_breaks,
                       source_units, stoplist_set, features_size):
    """Generate matching units based on unit information

    Parameters
    ----------
    source_units : list of dict
        each dictionary represents unit information from the source text
    target_units : list of dict
        each dictionary represents unit information from the target text
    stoplist_set : set of int
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
    dict [(int, int), 2d np.array of ints]
        see ``_bin_hits_to_unit_indices()`` for details on what this dictionary
        contains

    """
    # keep track of mapping between matrix row index and target unit index
    # in ``target_units``
    row2t_unit_ind = np.array([
        u_ind for u_ind in range(len(target_breaks) - 1)
        for _ in range(target_breaks[u_ind + 1] - target_breaks[u_ind])
    ])
    stepsize = 500
    for su_start in range(0, len(source_units), stepsize):
        search.update_current_stage_value(su_start / len(source_units))
        conn.update(search)
        feature_source_matrix, source_breaks = _construct_feature_unit_matrix(
            source_units[su_start:su_start + stepsize], stoplist_set,
            features_size)
        # for every position of each target unit, this matrix multiplication
        # picks up which source unit positions shared at least one common
        # feature
        match_matrix = target_feature_matrix.dot(feature_source_matrix)
        # this data structure keeps track of which target unit position matched
        # with which source unit position
        coo = match_matrix.tocoo()
        yield _bin_hits_to_unit_indices(coo.row, coo.col, row2t_unit_ind,
                                        target_breaks, source_breaks, su_start)


def _gen_matches(search, conn, target_units, source_units, stoplist_set,
                 features_size):
    """Generate match information where at least 2 positions matched

    Parameters
    ----------
    search : tesserae.db.entities.Search
        The search job associated with this matching job.
    conn : TessMongoConnection
    target_units : list of dict
        each dictionary represents unit information from the target text
    source_units : list of dict
        each dictionary represents unit information from the source text
    stoplist_set : set of int
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
    target_index : int
        index into ``target_units``
    source_index : int
        index into ``source_units``
    positions : 2d np.array
        the first column contains target positions; the second column has
        corresponding source positions
    """
    target_feature_matrix, target_breaks = _construct_unit_feature_matrix(
        target_units, stoplist_set, features_size)
    for hits2positions in gen_hits2positions(search, conn,
                                             target_feature_matrix,
                                             target_breaks, source_units,
                                             stoplist_set, features_size):
        overhits2positions = {
            k: np.array(v)
            for k, v in hits2positions.items() if len(v) >= 2
        }
        for (t_ind, s_ind), positions in overhits2positions.items():
            yield (t_ind, s_ind, positions)


def _score(search, conn, target_units, source_units, features, stoplist,
           distance_basis, max_distance, source_inv_frequencies_getter,
           target_inv_frequencies_getter, tag_helper):
    match_ents = []
    numerator_sparse_rows = []
    numerator_sparse_cols = []
    numerator_sparse_data = []
    denominators = []
    stoplist_set = set(stoplist)
    features_size = len(features)
    search_id = search.id
    for target_ind, source_ind, positions in _gen_matches(
            search, conn, target_units, source_units, stoplist_set,
            features_size):
        target_unit = target_units[target_ind]
        source_unit = source_units[source_ind]
        target_forms = np.array(target_unit['forms'])
        source_forms = np.array(source_unit['forms'])
        t_positions = positions[:, 0]
        s_positions = positions[:, 1]
        if distance_basis == 'span':
            # adjacent matched words have a distance of 2, etc.
            target_distance = _get_distance_by_span(t_positions, target_forms)
            source_distance = _get_distance_by_span(s_positions, source_forms)
        else:
            target_distance = _get_distance_by_least_frequency(
                target_inv_frequencies_getter, t_positions, target_forms)
            source_distance = _get_distance_by_least_frequency(
                source_inv_frequencies_getter, s_positions, source_forms)
        if source_distance <= 0 or target_distance <= 0:
            # less than two matching tokens in one of the units
            continue
        distance = source_distance + target_distance
        if distance <= max_distance:
            target_features = target_unit['features']
            source_features = source_unit['features']
            match_features = set(
                itertools.chain.from_iterable([
                    set(target_features[t_pos]).intersection(
                        set(source_features[s_pos]))
                    for t_pos, s_pos in zip(t_positions, s_positions)
                ]))
            match_features -= stoplist_set
            if match_features:
                match_inv_frequencies = [
                    target_inv_frequencies_getter(target_forms[pos])
                    for pos in set(t_positions)
                ]
                match_inv_frequencies.extend([
                    source_inv_frequencies_getter(source_forms[pos])
                    for pos in set(s_positions)
                ])
                numerator_sparse_rows.extend([len(match_ents)] *
                                             len(match_inv_frequencies))
                numerator_sparse_cols.extend(
                    [i for i in range(len(match_inv_frequencies))])
                numerator_sparse_data.extend(match_inv_frequencies)
                denominators.append(distance)
                match_ents.append(
                    Match(search_id=search_id,
                          source_unit=source_unit['_id'],
                          target_unit=target_unit['_id'],
                          source_tag=tag_helper.get_display_tag(
                              source_unit['text'], source_unit['tags']),
                          target_tag=tag_helper.get_display_tag(
                              target_unit['text'], target_unit['tags']),
                          matched_features=[
                              features[int(mf)].token for mf in match_features
                          ],
                          source_snippet=source_unit['snippet'],
                          target_snippet=target_unit['snippet'],
                          highlight=[
                              (int(s_pos), int(t_pos))
                              for s_pos, t_pos in zip(s_positions, t_positions)
                          ]))
    if match_ents:
        numerators = csr_matrix(
            (numerator_sparse_data, (numerator_sparse_rows,
                                     numerator_sparse_cols))).sum(axis=-1).A1
        scores = np.log(numerators) - np.log(denominators)
        for match, score in zip(match_ents, scores):
            match.score = score
    return match_ents
