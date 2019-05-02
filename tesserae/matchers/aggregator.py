import concurrent.futures
import multiprocessing as mp

import numpy as np
import pymongo

from tesserae.db import FeatureSet, Frequency, Match, MatchSet, Token


class AggregationMatcher(object):
    """Intertext matching using the Tesserae v3 similarity scoreself.

    Based on the similarity score described in [tess]_ .

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        Open connection to the Tesserae MongoDB instanceself.

    Attributes
    ----------
    connection : tesserae.db.TessMongoConnection
        Open connection to the Tesserae MongoDB instanceself.
    matches : list of tesserae.db.Match

    References
    ----------
    .. [tess] Forstall, C., Coffee, N., Buck, T., Roache, K., & Jacobson, S.
       (2014). Modeling the scholars: Detecting intertextuality through
       enhanced word-level n-gram matching. Digital Scholarship in the
       Humanities, 30(4), 503-515.
    """
    def __init__(self, connection):
        self.connection = connection
        self.clear()

    def clear(self):
        """Reset this Matcher to its initial state."""
        self.matches = []

    def get_stoplist(self, stopwords_list):
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
        pipeline = [{
            '$match': {
                'form': {'$in': stopwords_list}
            }
        }]
        stoplist = self.connection.aggregate(FeatureSet.collection, pipeline, encode=False)
        return [s['_id'] for s in stoplist]

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
        pipeline = []

        if basis != 'corpus':
            pipeline.append({
                '$match': {
                    'text': {'$in': [t.id for t in basis]}
                }
            })
        else:
            pipeline.extend([
                {'$lookup': {
                    'from': 'texts',
                    'localField': 'text',
                    'foreignField': '_id',
                    'as': 'text'
                }},
                {'$project': {
                    'frequency': True,
                    'feature_set': True,
                    'text': {'$arrayElemAt': ['$text', 0]}
                }},
                {'$match': {'text.language': language}}
            ])

        pipeline.extend([
            {'$group': {
                '_id': '$feature_set',
                'frequency': {'$sum': '$frequency'}
            }},
            {'$sort': {
                'frequency': -1
            }},
            {'$limit': n},
            {'$lookup': {
                'from': 'feature_sets',
                'let': {'fsid': '$_id'},
                'pipeline': [
                    {'$match': {'$expr': {'$eq': ['$_id', '$$fsid']}}},
                    {'$project': {'_id': False, feature: True}}
                ],
                'as': 'feature_set'
            }},
            {'$project': {
                'frequency': True,
                'feature': {'$arrayElemAt': ['$feature_set.' + feature, 0]}
            }},
            {'$unwind': '$feature'}
        ])

        stoplist = self.connection.aggregate('frequencies', pipeline, encode=False)
        return [s['_id'] for s in stoplist]

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
        stoplist = self.create_stoplist(
            stopwords,
            'form' if feature == 'form' else 'lemmata',
            texts[0].language,
            basis=stopword_basis)

        print(stoplist)
        import time
        start = time.time()

        pipeline = []

        match_set = MatchSet(
            texts=texts,
            unit_types=unit_type,
            feature=feature,
            parameters={
                'stopwords': stoplist,
                'stopword_basis': stopword_basis,
                'score_basis': score_basis,
                'frequency_basis': frequency_basis,
                'max_distance': max_distance,
                'distance_metric': distance_metric
            })
        self.connection.insert(match_set)

        match_texts = {
            '$match': {
                'text': {'$in': [t.id for t in texts]},
                'feature_set': {'$nin': stoplist + [None]}
            }
        }

        pipeline.append(match_texts)

        lookup_feature_set = \
            {'$lookup': {
                'from': 'feature_sets',
                'localField': 'feature_set',
                'foreignField': '_id',
                'as': 'feature_set'}}

        pipeline.append(lookup_feature_set)

        lookup_frequency = \
            {'$lookup': {
                'from': 'frequencies',
                'localField': 'frequency',
                'foreignField': '_id',
                'as': 'frequency'}}

        pipeline.append(lookup_frequency)

        reshape_relevant_fields = \
            {'$project': {
                'text': True,
                'index': True,
                'unit': '$' + unit_type,
                'feature': {'$arrayElemAt': ['$feature_set.' + feature, 0]},
                'frequency': {'$arrayElemAt': ['$frequency.frequency', 0]}
            }}

        pipeline.append(reshape_relevant_fields)

        pipeline.append({'$unwind': '$feature'})

        pipeline.append({'$sort': {'index': 1}})

        pipeline.append({'$group': {
            '_id': '$unit',
            'text': {'$addToSet': '$text'},
            'tokens': {'$push': '$_id'},
            'indices': {'$push': '$index'},
            'features': {'$push': '$feature'},
            'frequencies': {'$push': '$frequency'},
        }})

        pipeline.append({'$facet': {
            str(t.id): [
                {'$match': {'text': t.id}}
            ]
            for t in texts
        }})

        agg_matches = self.connection.aggregate('tokens', pipeline, encode=False)
        agg_matches = agg_matches.next()

        matches = []

        source = agg_matches[str(texts[0].id)]
        target = agg_matches[str(texts[1].id)]

        p = mp.Pool()
        results = p.map(score_wrapper, [(unit, target, distance_metric, max_distance, match_set) for unit in source])
        p.close()
        p.join()

        for r in results:
            matches.extend(r)
        min_score = np.min([match.score for match in matches])
        max_score = np.max([match.score for match in matches]) - min_score
        for match in matches:
            match.score = np.abs(np.ceil(10.0 * (match.score - min_score) / max_score))
            # if match.score > 10:
            #     match.score = 10.000

        return sorted(matches, key=lambda x: x.score, reverse=True), match_set


def score_wrapper(args):
    return score(*args)


def score(source_unit, target_units, distance_metric, max_distance, match_set):
    matches = []
    for t in target_units:
        intersect = set(source_unit['features']) & set(t['features'])
        if len(intersect) > 1:
            source_matches = np.array([i for i, f in enumerate(source_unit['features']) if f in intersect])
            target_matches = np.array([i for i, f in enumerate(t['features']) if f in intersect])

            source_indices = np.array(source_unit['indices'])[source_matches]
            target_indices = np.array(t['indices'])[target_matches]

            source_frequencies = np.array(source_unit['frequencies'], dtype=np.float32)[source_matches]
            target_frequencies = np.array(t['frequencies'], dtype=np.float32)[target_matches]

            if distance_metric == 'frequency':
                source_dist = source_indices[np.argsort(source_frequencies)[:1]]
                source_dist = np.abs(source_dist[1] - source_dist[0]) + 1
                target_dist = target_indices[np.argsort(target_frequencies)[:1]]
                target_dist = np.abs(target_dist[1] - target_dist[0]) + 1
            else:
                source_dist = np.abs(np.max(source_matches) - np.min(source_matches)) + 1
                target_dist = np.abs(np.max(target_matches) - np.min(target_matches)) + 1

            if source_dist < max_distance and target_dist <= max_distance:
                score = np.log(
                    (np.sum(np.power(source_frequencies, -1)) +
                     np.sum(np.power(target_frequencies, -1))) /
                    (source_dist + target_dist))
                matches.append(
                    Match(
                        units=[source_unit['_id'], t['_id']],
                        tokens=[source_unit['tokens'][i] for i in source_matches] +
                               [t['tokens'][i] for i in target_matches],
                        score=score,
                        match_set=match_set
                    ))
    return matches
