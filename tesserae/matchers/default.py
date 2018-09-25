import numpy as np

from tesserae.db import get_connection

# TODO: implement probabilistic stem matching once that's worked out


class DefaultMatcher(object):
    def __init__(self, connection):
        self.connection = connection

    def clear(self):
        """Reset this Matcher to its initial state."""
        self.matches = []

    def frequency_dist(self, match_tokens):
        """Compute distance based on frequency.

        The frequency distance computes the unmber of words separating the two
        match words with the lowest frequency in the unit.

        Parameters
        ----------
        match_tokens : list of (tesserae.db.Token, tesserae.db.Frequency)
            The tokens to compute the distance.

        Returns
        -------
        distance : float
            The number of words separating the two lowest-frequency tokens.
        """
        match_tokens.sort(key: x[1].frequency)
        return np.abs(match_tokens[0].index - match_tokens[1].index)

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
        pass

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
            formatted = {'corpus': {}}
            for f in frequencies:
                try:
                    formatted['corpus'][f.form] += f.frequency
                except KeyError:
                    formatted['corpus'][f.form] = f.frequency
        else:
            frequencies = self.collection.find('frequencies',
                                               text=[t.id for t in texts],
                                               form=[t.form for t in tokens])

            formatted = {t.cts_urn: {} for t in texts}
            for f in frequencies:
                formatted[f.text][f.form] = f.frequency

        return formatted

    def retrieve_lines(self, texts):
        """Get the lines associated with a text from the database.

        Parameters
        ----------
        text : tesserae.db.Text
            Text metadata.

        Returns
        -------
        lines : list of tesserae.db.Unit
            The line units in the order they appear in the original text.
        """
        lines = []

        for text in texts:
            lines.append(connection.find('units',
                         text=text.id, unit_type='line'))

        return lines

    def retrieve_phrases(self, texts):
        """Get the phrases associated with a text from the database.

        Parameters
        ----------
        text : tesserae.db.Text
            Text metadata.

        Returns
        -------
        phrases : list of tesserae.db.Unit
            The phrase units in the order they appear in the original text.
        """
        phrases = []

        for text in texts:
            phrases.append(connection.find('units',
                           text=text.id, unit_type='phrase'))

        return phrases

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

    def score(self, frequencies, distances):
        """
        """
        pass

    def span_dist(self, match_tokens):
        """Compute distance based on position in the text.

        The span distance computes the number of words separating the first and
        final match words in the unit.

        Parameters
        ----------
        match_tokens : list of (tesserae.db.Token, tesserae.db.Frequency)
            The tokens to compute the distance.

        Returns
        -------
        distance : float
            The number of words separating the two lowest-frequency tokens.
        """
        match_tokens.sort(key=lambda x: x[0].index)
        return np.abs(match_tokens[0].index, match_tokens[1].index)
