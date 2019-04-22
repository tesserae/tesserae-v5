import re
import unicodedata

from cltk.semantics.latin.lookup import Lemmata

from tesserae.tokenizers.base import BaseTokenizer


class GreekTokenizer(BaseTokenizer):
    def __init__(self, connection):
        super(GreekTokenizer, self).__init__(connection)

        # Set up patterns that will be reused
        self.vowels = 'αειηουωΑΕΙΗΟΥΩ'
        self.grave = '\u0300'
        self.acute = '\u0301'
        self.sigma = 'σ\b'
        self.sigma_alt = 'ς'
        self.word_characters = 'Ά-ώ' + self.sigma_alt + self.diacriticals

        self.diacrit_sub1 = \
            '([\s])([' + self.diacriticals + ']+)([' + self.vowels + ']{2,})'
        self.diacrit_sub2 = \
            '([\s])([' + self.diacriticals + ']+)([' + self.vowels + ']{1})'

        self.split_pattern = '([<].+[>])| / |[^\w' + self.diacriticals + self.sigma_alt + '\']'

        self.lemmatizer = Lemmata('lemmata', 'greek')

    def normalize(self, raw):
        """Normalize a single Greek word.

        Parameters
        ----------
        raw : str or list of str
            The word to normalize.

        Returns
        -------
        normalized : str
            The normalized string.
        """
        # Perform the global normalization
        normalized = super(GreekTokenizer, self).normalize(raw)

        # Convert grave accent to acute
        normalized = re.sub(self.grave, self.acute, normalized,
                            flags=re.UNICODE)

        # Remove diacriticals from vowels
        normalized = re.sub(self.diacrit_sub1, r'\1\3', normalized,
                            flags=re.UNICODE)
        normalized = re.sub(self.diacrit_sub2, r'\1\3\2', normalized,
                            flags=re.UNICODE)

        # Substitute sigmas
        normalized = re.sub(self.sigma, self.sigma_alt, normalized,
                            flags=re.UNICODE)

        normalized = re.sub(r'\'', '', normalized, flags=re.UNICODE)

        normalized = re.sub(r'[\'0-9]+', '', normalized,
                            flags=re.UNICODE)

        return normalized

    def featurize(self, tokens):
        """Get the features for a single Greek token.

        Parameters
        ----------
        token : str
            The token to featurize.

        Returns
        -------
        features : dict
            The features for the token.

        Notes
        -----
        Input should be sanitized with `greek_normalizer` prior to using this
        method.
        """
        features = []
        lemmata = self.lemmatizer.lookup(tokens)
        for i, l in enumerate(lemmata):
            features.append({'lemmata': [lem[0] for lem in l[1]]})
        return features
