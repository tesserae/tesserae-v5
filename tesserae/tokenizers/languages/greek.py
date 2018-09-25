import re
import unicodedata

from cltk.semantics.latin.lookup import Lemmata

from tesserae.tokenizers.languages.base import BaseTokenizer


class GreekTokenizer(BaseTokenizer):
    def __init__(self):
        super(GreekTokenizer, self).__init__()

        # Set up patterns that will be reused
        self.vowels = 'αειηουωΑΕΙΗΟΥΩ'
        self.grave = '\u0300'
        self.acute = '\u0301'
        self.sigma = 'σ\b'
        self.sigma_alt = 'ς'

        self.diacrit_sub1 = \
            '^([' + self.diacriticals + ']+)([' + self.vowels + ']{2,})'
        self.diacrit_sub2 = \
            '^([' + self.diacriticals + ']+)([' + self.vowels + ']{1})'

        self.split_pattern = '( / )|([^\w' + self.diacriticals + '\'])'

        self.lemmatizer = Lemmata('lemmata', 'greek')

    def normalize(self, tokens):
        """Normalize a single Greek word.

        Parameters
        ----------
        token : list of str
            The word to normalize.

        Returns
        -------
        normalized : str
            The normalized string.
        """
        # Perform the global normalization
        normalized = super(GreekTokenizer, self).normalize(tokens)

        # Convert grave accent to acute
        normalized = \
            [re.sub(self.grave, self.acute, n, flags=re.UNICODE)
             for n in normalized]

        # Remove diacriticals from vowels
        normalized = \
            [re.sub(self.diacrit_sub1, r'\2', n, flags=re.UNICODE)
             for n in normalized]
        normalized = \
            [re.sub(self.diacrit_sub2, r'\2\1', n, flags=re.UNICODE)
             for n in normalized]

        # Substitute sigmas
        normalized = \
            [re.sub(self.sigma, self.sigma_alt, n, flags=re.UNICODE)
             for n in normalized]

        normalized = \
            [re.sub(r'[\'0-9]+|[\s]+[0-9]+$', '', n, flags=re.UNICODE) for n in normalized]

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
            features.append({'lemmata': lemmata[i][1]})
        return features
