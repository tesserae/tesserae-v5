"""Unit matching by sparse matrix encoding.

Classes
-------

"""
import multiprocessing as mp

import numpy as np
from scipy.sparse import dok_matrix

from tesserae.db.entities import Entity, Feature, Match, MatchSet, Unit


class SparseMatrixSearch(object):
    def __init__(self, connection):
        self.connection = connection

    def get_stoplist(self, stopwords_list, language=None, feature=None):
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
                    'frequency': {
                        '$reduce': {
                            'input': {'$objectToArray': '$$frequencies'},
                            'initialValue': 0,
                            'in': {'$sum': ['$$value', '$$this.v']}
                        }
                    }
                }})
        else:
            basis = [t if not isinstance(t, Entity) else t.id for t in basis]
            pipeline.extend([
                {'$project': {
                    '_id': False,
                    'index': True,
                    'frequency': {'$sum': ['$frequencies.' + text for text in basis]}
                }}
            ])

        pipeline.extend([
            {'$sort': {'frequency': -1}},
            {'$limit': n},
            {'$project': {'index': True}}
        ])


        stoplist = self.connection.aggregate(Feature.collection, pipeline, encode=False)
        return np.array([s['index'] for s in stoplist], dtype=np.uint32)

    def match(self, texts, unit_type, feature, stopwords=10,
              stopword_basis='corpus', score_basis='word',
              frequency_basis='texts', max_distance=10,
              distance_metric='frequency'):
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
            stoplist = self.create_stoplist(
                stopwords,
                'form' if feature == 'form' else 'lemmata',
                texts[0].language,
                basis=stopword_basis)
        else:
            stoplist = get_stoplist(stopwords)

        match_matrices = []

        pipeline = [
            {'$match': {'feature': feature}},
            {'$count': 'count'}
        ]
        feature_count = self.collection.aggregate(Feature.collection, pipeline)
        feature_count = feature_count['count']

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
                    '_id': False,
                    'index': True,
                    'tokens': True
                }},
                # Create one document per token, preserving unit data
                # {{'$unwind': 'tokens'}},
                # Look up the intended features for eahc token
                {'$lookup': {
                    'from': 'tokens',
                    'let': {'t_id': '$tokens'},
                    'pipeline': [
                        {'$match': {'$exec': {'$eq': ['$_id', '$$t_id']}}},
                        {'$project': {
                            '_id': False,
                            'token_index': True,
                            'feature': '$feature.' + feature
                        }},
                        {'$lookup': {
                            'from': 'features',
                            'let': {'fids': '$tokens'},
                            'pipeline': [
                                {'$match': {'$expr': {'$in': ['_id', '$$fids']}}},
                                {'$project': {
                                    '_id': False,
                                    'index': True,
                                    'token': True,
                                    'frequency': '$frequencies.' + t.id
                                }}
                            ],
                            'as': 'feature'
                        }}
                    ],
                    'as': 'tokens'
                }}
            ]

            # The returned documents should look like:
            # {
            #      index: <int>,             # Unit index in text
            #      tokens: [{
            #          token_index: <int>,   # Token index in text
            #          feature: {
            #              index: <int>,     # Sparse matrix index of feature
            #              token: <str>,     # String representation of feature
            #              frequency: <int>  # Frequency of the feature
            #          }
            #      }]
            # }

            units = list(self.connection.aggregate(
                Unit.collection, pipeline, encode=False))

            unit_indices = []
            feature_indices = []

            for unit in units:
                unit_index = unit['index']
                for token in unit['tokens']:
                    feature = token['feature']
                    if isinstance(feature['token'], str):
                        unit_indices.append(unit_index)
                        feature_indices.append(token['feature']['index'])
                    else:
                        unit_indices.extend([unit_index for _ in range(len(feature['token']))])
                        feature_indices.extend(feature['index'])

            feature_matrix = dok_matrix((unit_indices[-1], feature_count))
            feature_matrix[(np.asarray(unit_indices), np.asarray(feature_indices))] = 1

            del unit_indices, feature_indices

            unit_matrices.append(feature_matrix)
            unit_lists.append(units)

        matches = np.nonzero(np.matmul(unit_matrices[1], unit_matrices[0].T) > 1)

        print(matches)
