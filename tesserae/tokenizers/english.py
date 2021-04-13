import re

import nltk.corpus.reader.wordnet
from nltk.corpus import wordnet
from tesserae.tokenizers.base import BaseTokenizer


class EnglishTokenizer(BaseTokenizer):
    def __init__(self, connection):
        super(EnglishTokenizer, self).__init__(connection)

        self.word_regex = re.compile('\'?[a-zA-Z]+(?:[\'\\-][a-zA-Z]*)?',
                                     flags=re.UNICODE)

        self.split_pattern = \
            '( / )|([\\s]+)|([^\\-\'\\w\\d' + self.diacriticals + '])|' + \
            '([\']+(?![a-zA-Z]+))|((?<![a-zA-Z])-(?![a-zA-Z]+))|(--+)'

    def normalize(self, raw, split=True):
        """Normalize an English word.

        Parameters
        ----------
        raw : str or list of str
            The string(s) to normalize.

        Returns
        -------
        normalized : str or list of str
            The normalized string(s).

        Notes
        -----
        This function should be applied to English words prior to generating
        other features (e.g., lemmata).

        """
        # Apply the global normalizer
        normalized, tags = super(EnglishTokenizer, self).normalize(raw)

        if split:
            normalized = re.split(self.split_pattern,
                                  normalized,
                                  flags=re.UNICODE)
            normalized = [
                t for t in normalized if t and self.word_regex.search(t)
            ]

        return normalized, tags

    def featurize(self, tokens):
        """Lemmatize English tokens.

        Parameters
        ----------
        tokens : list of str
            The tokens to featurize.

        Returns
        -------
        result : dict
            The features for the tokens. Every feature type should be listed as
            a key. For every key, the value should be a list of lists. Every
            position in the outer list corresponds to the same position in the
            tokens list passed in. Every inner list contains all instances of
            the feature type associated with the corresponding token.

        Example
        -------
        Suppose we have the following tokens to featurize:
        >>> tokens = ['rung', 'defeated']

        Then the result would look something like this:
        >>> result = {
        >>>     'lemmata': [
        >>>         ['rung', 'ring'],
        >>>         ['defeated', 'defeat']
        >>>     ]
        >>> }

        Note that `result['lemmata'][0]` is a list containing the lemmata for
        `tokens[0]`; similarly, `result['lemmata'][1]` is a list containing the
        lemmata for `tokens[1]`.

        Notes
        -----
        Input should be sanitized with `EnglishTokenizer.normalize` prior to
        using this method.

        """
        if not isinstance(tokens, list):
            tokens = [tokens]
        lemmata = []
        for token in tokens:
            lemma_set = set()
            for pos in nltk.corpus.reader.wordnet.POS_LIST:
                lemma_set.update(wordnet._morphy(token, pos))
            if lemma_set:
                lemmata.append(sorted([lemma for lemma in lemma_set]))
            else:
                lemmata.append([token])
        features = {'lemmata': lemmata}
        return features
