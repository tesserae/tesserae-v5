import concurrent.futures

import pymongo

from tesserae.db import Match, MatchSet


class DefaultMatcher(object):
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

    def create_stoplist(self, n, feature, basis=None):
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

        if basis is not None:
            pipeline.append(
                {'$match': {'text': {'$in': [t.id for t in basis]}}})

        pipeline.extend([
            {'$group': {'_id': '$form', 'frequency': {'$push': '$frequency'}}},
            {'$project': {'form': 1, 'frequency': {'$sum': '$frequency'}}},
            {'$sort': {'frequency': -1}},
            {'$limit': n}
        ])

        stoplist = self.connection.aggregate('frequencies', pipeline)

        return stoplist

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
        stoplist = self.create_stoplist(stopwords, basis=stopword_basis)

        pipeline = []

        match_on_text = \
            {'$match': {
                'text': {'$in': [t.id for t in texts]},
                'feature_set': {'$ne': None}}}

        lookup_frequencies = \
            {'$lookup': {
                'from': 'feature_sets',
                'localField': 'feature_set',
                'foreignField': '_id',
                'as': 'features'}}

        remove_stopwords = \
            {'$match': {
                'features.' + feature: {'$nin': stoplist + [None]}}}

        reshape_relevant_fields = \
            {'$project': {
                'text': True,
                'index': True,
                'unit': '$' + unit_type,
                'feature': {'$arrayElemAt': ['$features.' + feature, 0]},
                'frequency': {'$arrayElemAt': ['$features.frequency', 0]}}}

        flatten_features = \
            {'$unwind': '$feature'}

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
                            'frequency': '$frequency.' + str(t.id)
                        },
                        None]}}
             for t in texts})

        remove_null_entries = \
            {'$project': {
                str(t.id): {
                    '$setDifference': ['$' + str(t.id), [None]]}
                for t in texts}}

        flatten_by_text = [{'$unwind': '$' + str(t.id)} for t in texts]

        group_by_common_unit = \
            {'$group': {
                '_id': {str(t.id): '$' + str(t.id) + '.unit' for t in texts}}}
        group_by_common_unit['$group'].update(
            {str(t.id): {'$push': '$' + str(t.id)} for t in texts})

        determine_if_match = \
            {'$project': {
                str(t.id): True,
                str(t.id) + '_arraySize': {'$size': '$' + str(t.id)}}
             for t in texts}

        remove_non_matches = \
            {'$match': {str(t.id) + '_arraySize': {'$gt': 1} for t in texts}}

        sort_matches = {'$sort': {'score': -1}}

        pipeline = [match_on_text, lookup_frequencies, remove_stopwords,
                    reshape_relevant_fields, flatten_features,
                    group_by_feature, remove_null_entries, flatten_by_text,
                    group_by_common_unit, determine_if_match,
                    remove_non_matches]

        agg_matches = self.connection.aggregate('tokens', pipeline)

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

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {
                executor.submit(self.score, match_doc, texts): match_doc
                for match_doc in agg_matches}
            for future in concurrent.futures.as_completed(future_to_url):
                matches.append(future_to_match[future])

        self.matches = matches

        return matches

    def score(self, match_doc, texts, distance_metric):
        frequencies = np.array([[token.frequency for token in match_doc[str(text.id)]] for text in texts])

        if distance_metric == 'frequency':
            idx = np.argsort(frequencies, axis=-1)
            distances = np.abs(idx[:, 1] - idx[:, 0])
        else:
            indices = np.array([[token.index for token in match_doc[str(text.id)]] for text in texts])
            distances = np.abs(np.max(indices, axis=-1) - np.min(indices, axis=-1))

        return np.ln(np.sum(np.pow(np.sum(frequencies, axis=-1), -1.0)), np.sum(distances))
