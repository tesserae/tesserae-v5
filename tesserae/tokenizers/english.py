import re

from nltk.stem import WordNetLemmatizer

from tesserae.tokenizers.base import BaseTokenizer


class EnglishTokenizer(BaseTokenizer):
    def __init__(self, connection):
        super(EnglishTokenizer, self).__init__(connection)

        self.lemmatizer = WordNetLemmatizer()

        self.split_pattern = \
            r'[^\w]+'

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
            normalized = re.split(self.split_pattern, normalized, flags=re.UNICODE)
            normalized = [t for t in normalized if t]

        return normalized, tags

    def featurize(self, tokens):
        """Lemmatize an English token.

        Parameters
        ----------
        tokens : list of str
            The token to featurize.

        Returns
        -------
        lemmata : dict
            The features for the token.

        Notes
        -----
        Input should be sanitized with `EnglishTokenizer.normalize` prior to
        using this method.

        """
        if not isinstance(tokens, list):
            tokens = [tokens]
        features = {
            'lemmata': [self.lemmatizer.lemmatize(token) for token in tokens]
        }
        return features
