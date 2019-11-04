"""Unit matching by sparse matrix encoding.

Classes
-------

"""
import itertools
import multiprocessing as mp

import numpy as np
import pymongo
from scipy.sparse import dok_matrix

from tesserae.db.entities import Entity, Feature, Match, MatchSet, Unit


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

    def get_frequencies(self, feature, language, basis='corpus', count=None):
        """Get frequency data for a given feature.
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

        freqs = self.connection.aggregate(Feature.collection, pipeline, encode=False)
        freqs = list(freqs)
        freqs = np.array([freq['frequency'] for freq in freqs])
        return freqs / sum(freqs)

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

        frequency_basis = frequency_basis if frequency_basis != 'texts' \
                          else texts

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

        frequencies = self.get_frequencies(feature, texts[0].language, basis=frequency_basis)
        features = sorted(self.connection.find('features', language=texts[0].language, feature=feature), key=lambda x: x.index)

        match_ents = []
        match_set = MatchSet(texts=texts)
        # with mp.Pool() as p:
        #    result = p.map(score_wrapper, [(source_unit, units[1], frequencies, distance_metric, max_distance, -np.inf) for source_unit in units[0]])
        for i in range(matches.shape[0]):
            source_unit = unit_lists[0][i]
            target_units = [unit_lists[1][j] for j in matches[i].nonzero()[1]]
            match_ents.extend(score(source_unit, target_units, frequencies, features, distance_metric, max_distance, min_score, match_set))



        return match_ents, match_set


def score_wrapper(args):
    return score(*args)


def score(source, targets, frequencies, features, distance_metric, maximum_distance, min_score, match_set):
    matches = []
    source_features = list(itertools.chain.from_iterable(source['features'][0]['features']))  # list(np.array(source['features'][0]['features']).ravel())
    # print(source['features'][0])
    source_indices = list([[source['features'][0]['index'][i] for _ in range(len(source['features'][0]['features'][i]))] for i in range(len(source['features'][0]['index']))])
    source_indices = list(itertools.chain.from_iterable(source_indices))  # list(np.array(source_indices).ravel())
    source_frequencies = np.array([frequencies[i] for i in source_features])

    for target in targets:
        target_features = list(itertools.chain.from_iterable(target['features'][0]['features']))  # list(np.array(target['features'][0]['features']).ravel())
        target_indices = list([[target['features'][0]['index'][i] for _ in range(len(target['features'][0]['features'][i]))] for i in range(len(target['features'][0]['index']))])
        target_indices = list(itertools.chain.from_iterable(target_indices))  # list(np.array(target_indices).ravel())
        target_frequencies = np.array([frequencies[i] for i in target_features])

        # print(source_features, target_features)
        match_features = set(source_features).intersection(set(target_features))
        match_idx = [source_features.index(i) for i in match_features]
        match_idx = [source_indices[i] for i in match_idx]

        if not match_idx.count(match_idx[0]) == len(match_idx) and match_idx:
            match_frequencies = [frequencies[i] for i in match_features]

            if distance_metric == 'span':
                source_idx = np.array([source_indices[i] for i in [source_features.index(f) for f in match_features]])
                target_idx = np.array([target_indices[i] for i in [target_features.index(f) for f in match_features]])
                source_distance = np.abs(np.max(source_idx) - np.min(source_idx)) + 1
                target_distance = np.abs(np.max(target_idx) - np.min(target_idx)) + 1
            else:
                freq_sort = np.argsort(source_frequencies)
                # print(freq_sort.shape[0], len(source_features), len(source_indices))
                source_idx = np.array([source_indices[i] for i in freq_sort])
                # try:
                if source_idx.shape[0] > 2:
                    s_end = [s for s in source_idx if s != source_idx[0]][0]
                else:
                    continue
                # except IndexError:
                #     s_end = 1
                freq_sort = np.argsort(target_frequencies)
                target_idx = np.array([target_indices[i] for i in freq_sort])
                # try:
                if target_idx.shape[0] > 2:
                    t_end = [t for t in target_idx if t != target_idx[0]][0]
                else:
                    continue
                # except IndexError:
                #     t_end = 1
                source_distance = np.abs(s_end - source_idx[0]) + 1
                target_distance = np.abs(t_end - target_idx[0]) + 1

            score = -np.inf
            if source_distance < maximum_distance and target_distance < maximum_distance:
                score = np.log((np.power(np.sum(match_frequencies), -1) * 2) / (source_distance + target_distance))
                # print('Score: {}'.format(score))

            if score >= min_score:
                # print('Adding match')
                matches.append(
                    Match(
                        units=[source['_id'], target['_id']],
                        tokens=[features[int(mf)] for mf in match_features],
                        score=score,
                        match_set=match_set))

    return matches
