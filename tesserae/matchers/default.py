import numpy as np

from tesserae.db import get_connection, Match

# TODO: implement probabilistic stem matching once that's worked out


class DefaultMatcher(object):
    def __init__(self, connection):
        self.connection = connection

    def clear(self):
        """Reset this Matcher to its initial state."""
        self.matches = []

    def frequency_dist(self, frequency_vector):
        """Compute distance based on frequency.

        The frequency distance computes the unmber of words separating the two
        match words with the lowest frequency in the unit.

        Parameters
        ----------
        frequency_vector : list of float
            The frequencies of the tokens.

        Returns
        -------
        distance : float
            The number of words separating the two lowest-frequency tokens.
        """
        ordered = np.argsort(distance_vector, axis=-1, kind='heapsort')
        return np.abs(ordered[1] - ordered[0]) + 1

    def match(texts, unit_type, stopwords=10, stopword_basis='corpus',
              score_basis='word', frequency_basis='texts', max_distance=10,
              distance_metric='frequency'):
        """Find matches between one or more texts.

        Texts will contain lines or phrases with matching tokens, with varying
        degrees of strength to the match. If one text is provided, each unit in
        the text will be matched with every subsequent unit.

        Parameters
        ----------
        texts : list of tesserae.db.Text
            The texts to match. Texts are matched in
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
        tokens = self.retrieve_tokens(texts)
        units = self.retreve_units(texts, unit_type)
        frequencies = self.retrieve_frequencies(texts, tokens, frequency_basis)

        # TODO: recursive scheme for matching
        # matches = find_matches(units, score_basis, tokens, 0, 1)

        matches = []

        for unit_a in units[0]:
            for unit_b in units[1]:
                match = Match(units=[unit_a, unit_b])
                match_tokens = []
                distance_vector = [[], []]
                match_frequencies = [[], []]
                tokens_a = [tokens[t] for t in unit_a.tokens]
                tokens_b = [tokens[t] for t in unit_b.tokens]
                for token_a in tokens_a:
                    if distance_metric == 'frequency':
                        distance_vector[0].append(
                            frequencies[token_a.form].frequency)
                    for token_b in tokens_b:
                        if distance_metric == 'frequency':
                            distance_vector[1].append(
                                frequencies[token_b.form].frequency)
                        if token_a.match(token_b, feature):
                            match_tokens.append((token_a, token_b))
                            if distance_metric == 'span':
                                distance_vector[0].append(token_a.index)
                                distance_vector[1].append(token_b.index)

                dist = map(self.compute_distance, distance_vector)

                if len(match_tokens) >= 2 and all(dist < max_distance):
                    dist = sum(dist)
                    freq = sum(match_frequencies)
                    match.score = math.log(freq / dist)
                    match.match_tokens = match_tokens
                    matches.append(match)

        self.matches = matches
        return matches

    def retrieve_frequencies(self, texts, tokens, basis):
        """Get token frequencies for the tokens to be matched.

        Parameters
        ----------
        texts : 'corpus' or list of tesserae.db.Text
            The texts from which to compute the token frequencies. If 'corpus',
            use token frequencies from the entire collection of texts in the
            database. Otherwise, use frequencies within the specified texts.

        tokens : list of tesserae.db.Token
            The tokens that should be included in the match.
        basis : {'word', 'stem'}


        Returns
        -------
        frequencies : dict
            Key/value pairs of frequencies indexed by text and token string:
            {{
                <text_id>: {{
                    <token_string>: <frequency>
                }}
            }}
        """
        if basis == 'corpus':
            frequencies = self.collection.find('frequencies',
                                               form=[t.form for t in tokens])
        else:
            frequencies = self.collection.find('frequencies',
                                               text=[t.id for t in texts],
                                               form=[t.form for t in tokens])

        formatted = {}
        for f in frequencies:
            try:
                formatted[f.form] += f.frequency
            except KeyError:
                formatted[f.form] = f.frequency

        return formatted

    def retrieve_tokens(self, texts):
        """Get the tokens associated with a text from the database.

        Parameters
        ----------
        text : tesserae.db.Text
            Text metadata.

        Returns
        -------
        tokens : list of tesserae.db.Token
            The tokens in the order they appear in the original text.
        """
        tokens = []

        for text in texts:
            tokens.append(connection.find('units', text=text.id))

        return phrases

    def retrieve_units(self, texts, unit_type):
        """Get the units associated with a text from the database.

        Parameters
        ----------
        text : tesserae.db.Text
            Text metadata.
        unit_type : {'line','phrase'}
            The type of unit to retrieve.

        Returns
        -------
        units : list of tesserae.db.Unit
            The units in the order they appear in the original text.
        """
        units = []

        for text in texts:
            units.append(connection.find('units',
                         text=text.id, unit_type=unit_type))

        return units

    def span_distance(self, index_vector):
        """Compute distance based on position in the text.

        The span distance computes the number of words separating the first and
        final match words in the unit.

        Parameters
        ----------
        index_vector : list of int
            The indices of the match tokens.

        Returns
        -------
        distance : float
            The number of words separating the two lowest-frequency tokens.
        """
        ordered = np.argsort(index_vector, axis=-1, kind='heapsort')
        return np.abs(index_vector[ordered[0]] - index_vector[ordered[-1]]) + 1
