import concurrent.futures

import numpy as np
import pymongo

from tesserae.db import Frequency, Match, MatchSet, Token


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
        stoplist : list of str
            The `n` most frequent tokens in the basis texts.
        """
        pipeline = []

        if basis != 'corpus':
            print("Usint texts")
            pipeline.append({
                '$match': {
                    'text': {'$in': [t.id for t in basis]}
                }
            })
        else:
            print("Using corpus")
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
            {'$unwind': '$feature'},
            {'$sort': {
                'frequency': -1
            }},
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

        pipeline = []

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

        # remove_stopwords = \
        #     {'$match': {
        #         'feature': {'$nin': stoplist + [None]}}}
        #
        # pipeline.append(remove_stopwords)

        group_by_feature = \
            {'$group': {'_id': '$feature'}}
        group_by_feature['$group'].update(
            {str(t.id): {
                '$push': {
                    '$cond': [
                        {'$eq': ['$text', t.id]},
                        {
                            '_id': '$_id',
                            'unit': '$unit',
                            'feature': '$feature',
                            'index': '$index',
                            'frequency': '$frequency'
                        },
                        None]}}
             for t in texts})

        pipeline.append(group_by_feature)

        remove_null_entries = \
            {'$project': {
                str(t.id): {
                    '$setDifference': ['$' + str(t.id), [None]]}
                for t in texts}}
        remove_null_entries['$project'].update(
                {str(t.id) + '_arraySize': {
                    '$size': {
                        '$setDifference': ['$' + str(t.id), [None]]}}
                for t in texts})

        pipeline.append(remove_null_entries)

        remove_null_entries = \
            {'$match': {str(t.id) + '_arraySize': {'$gte': 1} for t in texts}}

        pipeline.append(remove_null_entries)

        flatten_by_text = [{'$unwind': '$' + str(t.id)} for t in texts]

        pipeline.extend(flatten_by_text)

        group_by_common_unit = \
            {'$group': {
                '_id': {str(t.id): '$' + str(t.id) + '.unit' for t in texts}}}
        group_by_common_unit['$group'].update(
            {str(t.id): {'$push': '$' + str(t.id)} for t in texts})

        pipeline.append(group_by_common_unit)

        determine_if_match = \
            {'$project': {str(t.id): True for t in texts}}
        determine_if_match['$project'].update({
            str(t.id) + '_arraySize': {'$size': '$' + str(t.id)}
            for t in texts})

        pipeline.append(determine_if_match)

        remove_non_matches = \
            {'$match': {str(t.id) + '_arraySize': {'$gt': 1} for t in texts}}

        pipeline.append(remove_non_matches)

        # pipeline = [match_on_text, lookup_frequencies, remove_stopwords,
        #             reshape_relevant_fields, flatten_features,
        #             group_by_feature, remove_null_entries]
        # pipeline.extend(flatten_by_text)
        # pipeline.extend([group_by_common_unit, determine_if_match,
        #                  remove_non_matches])

        agg_matches = self.connection.aggregate('tokens', pipeline, encode=False)
        # print(agg_matches[:20])
        # return [], None

        match_set = MatchSet(
            texts=texts,
            unit_type=unit_type,
            feature=feature,
            parameters={
                'stopwords': stoplist,
                'stopword_basis': stopword_basis,
                'score_basis': score_basis,
                'frequency_basis': frequency_basis,
                'max_distance': max_distance,
                'distance_metric': distance_metric
            })

        matches = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_match = {
                executor.submit(self.score, match_doc, texts, distance_metric, match_set): match_doc
                for match_doc in agg_matches}
            for future in concurrent.futures.as_completed(future_to_match):
                matches.append(future.result())

        self.matches = matches

        return sorted(matches, key=lambda x: x.score, reverse=True), match_set


    def score(self, match_doc, texts, distance_metric, match_set=None):
        frequencies = np.array([[token['frequency'] for token in match_doc[str(t.id)]] for t in texts])

        if distance_metric == 'frequency':
            idx = np.argsort(frequencies, axis=-1)
            distances = np.abs(idx[:, 1] - idx[:, 0])
        else:
            indices = np.array([[token['index'] for token in match_doc[str(t.id)]] for t in texts])
            distances = np.abs(np.max(indices, axis=-1) - np.min(indices, axis=-1))

        score = np.log(np.sum(np.power(np.sum(frequencies, axis=-1), -1.0)) / np.sum(distances))
        match = Match(
            units=list(match_doc['_id'].values()),
            tokens=[
                [match_doc[str(t.id)][i]['_id'] for i in range(len(match_doc[str(t.id)]))]
                for t in texts],
            score=score,
            match_set=match_set
        )
        return match
