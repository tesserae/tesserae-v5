"""Unit matching by sparse matrix encoding.

Classes
-------

"""
import itertools
import multiprocessing as mp

from bson import ObjectId
import numpy as np
import pymongo
from scipy.sparse import dok_matrix

from tesserae.db.entities import Entity, Feature, Match, MatchSet, Token, Text, Unit


class SparseMatrixSearch(object):
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
            basis = [t if not isinstance(t, Entity) else str(t.id) for t in basis]
            pipeline.extend([
                {'$project': {
                    '_id': False,
                    'index': True,
                    'token': True,
                    'frequency': {
                        '$sum': ['$frequencies.' + text for text in basis]}
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

    def get_frequencies(self, feature, language, basis='corpus'):
        """Get frequency data for a given feature.

        Frequency is equal to the number of times a feature occurs in the basis
        divided by the total number of tokens in the basis.

        Note that the sum of all features is not equivalent to the sum of all
        tokens, since every token can have multiple instances of the same
        feature type associated with it.
        """
        freqs = _get_feature_counts(self.connection, feature, language, basis)
        token_count = _get_token_count(self.connection, language, basis)
        return freqs / token_count

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

        pipeline = [
            {'$match': {'feature': feature}},
            {'$count': 'count'}
        ]
        feature_count = self.connection.aggregate(Feature.collection, pipeline, encode=False)
        feature_count = next(feature_count)['count']

        unit_matrices = []
        unit_lists = []

        for t in texts:
            pipeline = [
                # Get the units from text t with the specified unit type
                {'$match': {
                    'text': t.id,
                    'unit_type': unit_type
                }},
                # Convert the tokens to their features
                {'$project': {
                    'index': True,
                    'features': [{
                        'index': '$tokens.index',
                        'features': '$tokens.features.' + feature
                    }]
                }},
            ]

            units = list(self.connection.aggregate(
                Unit.collection, pipeline, encode=False))

            unit_indices = []
            feature_indices = []

            for unit in units:
                if 'features' in unit:
                    all_features = list(itertools.chain.from_iterable(unit['features'][0]['features'])) # list(np.array(unit['features'][0]['features']).ravel())
                    unit_indices.extend([unit['index'] for _ in range(len(all_features))])
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


        if frequency_basis != 'texts':
            source_frequencies = self.get_frequencies(feature,
                    texts[0].language, basis=frequency_basis)
            target_frequencies = source_frequencies
        else:
            source_frequencies = self.get_frequencies(feature,
                    texts[0].language, basis=[texts[0]])
            target_frequencies = self.get_frequencies(feature,
                    texts[1].language, basis=[texts[1]])
        features = sorted(self.connection.find('features', language=texts[0].language, feature=feature), key=lambda x: x.index)
        stoplist_set = set(stoplist)

        match_ents = []
        match_set = MatchSet(texts=texts)
        # with mp.Pool() as p:
        #    result = p.map(score_wrapper, [(source_unit, units[1], frequencies, distance_metric, max_distance, -np.inf) for source_unit in units[0]])
        for i in range(matches.shape[0]):
            source_unit = unit_lists[0][i]
            target_units = [unit_lists[1][j] for j in matches[i].nonzero()[1]]
            match_ents.extend(score(source_unit, target_units,
                source_frequencies, target_frequencies,
                features, stoplist_set, distance_metric, max_distance, min_score, match_set))



        return match_ents, match_set


def score_wrapper(args):
    return score(*args)


def score(source, targets, in_source_frequencies, in_target_frequencies,
        features, stoplist_set, distance_metric, maximum_distance, min_score, match_set):
    matches = []
    '''
    ``source`` is a dictionary with the following keys:
        _id: matches a Unit.id in the database representing a unit in the
            source text
        index: an int corresponding to the index of the Unit with same id as
            _id
        features: a list containing a single dictionary

    the dictionary under ``source['features']`` has the following keys:
        index: a list of ints corresponding to Token indices
        features: a list of lists; each position in the outer list corresponds
            to the same position in ``source['features'][0]['index']``; the
            inner list contains ints corresponding to Feature indices; the type
            of Feature being referred to was determined by the match() call
    '''
    # print(source)
    '''
    ``source_features`` is a flattened version of
    ``source['features'][0]['features'] (i.e., the list of lists has become a
    list)
    '''
    source_features = list(itertools.chain.from_iterable(source['features'][0]['features']))  # list(np.array(source['features'][0]['features']).ravel())
    source_features_set = set(source_features)
    '''
    ``source_indices[i]`` is equal to the token position in the unit to which
    ``source_features[i]`` corresponds
    '''
    source_indices = list([[i for _ in range(len(source['features'][0]['features'][i]))] for i in range(len(source['features'][0]['index']))])
    source_indices = list(itertools.chain.from_iterable(source_indices))  # list(np.array(source_indices).ravel())
    source_frequencies = np.array([in_source_frequencies[i] for i in source_features])

    for target in targets:
        target_features = list(itertools.chain.from_iterable(target['features'][0]['features']))  # list(np.array(target['features'][0]['features']).ravel())
        target_indices = list([[i for _ in range(len(target['features'][0]['features'][i]))] for i in range(len(target['features'][0]['index']))])
        target_indices = list(itertools.chain.from_iterable(target_indices))  # list(np.array(target_indices).ravel())
        target_frequencies = np.array([in_target_frequencies[i] for i in target_features])

        # print(source_features, target_features)
        match_features = \
                source_features_set.intersection(
                        set(target_features)).difference(stoplist_set)
        match_idx = [source_features.index(i) for i in match_features]
        match_idx = [source_indices[i] for i in match_idx]

        if not match_idx.count(match_idx[0]) == len(match_idx) and match_idx:
            match_frequencies = \
                    [in_target_frequencies[i] for i in match_features] + \
                    [in_source_frequencies[i] for i in match_features]

            print('target frequencies', [(in_target_frequencies[i], i) for i in match_features])
            print('target positions', target['features'][0]['features'])
            print('source frequencies', [(in_source_frequencies[i], i) for i in match_features])
            print('source positions', source['features'][0]['features'])

            if distance_metric == 'span':
                source_idx = np.array([source_indices[i] for i in [source_features.index(f) for f in match_features]])
                target_idx = np.array([target_indices[i] for i in [target_features.index(f) for f in match_features]])
                source_distance = np.abs(np.max(source_idx) - np.min(source_idx))
                target_distance = np.abs(np.max(target_idx) - np.min(target_idx))
            else:
                source_distance = _get_distance_by_least_frequency(
                        source_frequencies,
                        source_indices, source_features, match_features)
                target_distance = _get_distance_by_least_frequency(
                        target_frequencies,
                        target_indices, target_features, match_features)
                if source_distance <= 0 or target_distance <= 0:
                    continue

            score = -np.inf
            if source_distance < maximum_distance and target_distance < maximum_distance:
                score = np.log((np.sum(np.power(match_frequencies, -1))) / (source_distance + target_distance))
                # print('Score: {}'.format(score))

            print(score, target_distance, source_distance)
            if score >= min_score:
                # print('Adding match')
                matches.append(
                    Match(
                        units=[source['_id'], target['_id']],
                        tokens=[features[int(mf)] for mf in match_features],
                        score=score,
                        match_set=match_set))

    return matches


def _get_feature_counts(connection, feature, language, basis):
    """
    """
    pipeline = [
        {'$match': {'feature': feature, 'language': language}}
    ]

    if basis == 'corpus':
        pipeline.append(
            {'$project': {
                '_id': False,
                'index': True,
                'frequency': {
                    '$reduce': {
                        'input': {'$objectToArray': '$frequencies'},
                        'initialValue': 0,
                        'in': {'$sum': ['$$value', '$$this.v']}
                    }
                }
            }})
    else:
        basis = [t if not isinstance(t, Entity) else str(t.id) for t in basis]
        pipeline.extend([
            {'$project': {
                '_id': False,
                'index': True,
                'token': True,
                'frequency': {'$sum': ['$frequencies.' + text for text in basis]}
            }}
        ])

    pipeline.extend([
        {'$sort': {'index': 1}}
    ])

    freqs = connection.aggregate(Feature.collection, pipeline, encode=False)
    freqs = list(freqs)
    freqs = np.array([freq['frequency'] for freq in freqs])
    return freqs


def _get_token_count(connection, language, basis):
    if basis == 'corpus':
        text_ids = [ObjectId(t.id) for t in connection.find(Text.collection,
            language=language)]
    else:
        text_ids = [ObjectId(t.id) if isinstance(t, Entity) else ObjectId(t) for t in basis]
    return connection.connection[Token.collection].count_documents(
            {'text': {'$in': text_ids},
                # https://stackoverflow.com/a/6838057
                'features': {'$gt': {}}})


def _get_distance_by_least_frequency(frequencies, indices, features,
        match_features):
    matched_frequencies = [
            frequencies[i]
            for i, feature in enumerate(features)
            if feature in match_features]
    matched_indices = [
            indices[i]
            for i, feature in enumerate(features)
            if feature in match_features]
    freq_sort = np.argsort(matched_frequencies)
    idx = np.array([matched_indices[i] for i in freq_sort])
    if idx.shape[0] >= 2:
        end = [s for s in idx if s != idx[0]][0]
        return np.abs(end - idx[0])
    return 0
