from tesserae.db import get_connection

# TODO: implement probabilistic stem matching once that's worked out


class DefaultMatcher(object):
    def __init__(self, connection):
        self.connection = connection

    def clear(self):
        """Reset this Matcher to its initial state."""
        self.matches = []

    def frequency_dist(self, match_tokens):
        pass

    def match(texts, stopwords=10, stopword_basis='corpus', score_basis='word',
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
        """
        """
        pass

    def retrieve_lines(self, text):
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
        pass

    def retrieve_phrases(self, text):
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
        pass

    def retrieve_tokens(self, text):
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
        pass

    def score(self, frequencies, distances):
        """
        """
        pass

    def span_dist(self, tokens, match_tokens):
        """
        """
        pass
