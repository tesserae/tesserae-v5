import re

from cltk.semantics.latin.lookup import Lemmata

from tesserae.tokenizers.languages.base import BaseTokenizer


class GreekTokenizer(BaseTokenizer):
    def __init__(self):
        # Set up patterns that will be reused
        self.diacriticals = re.compile(
            '[\u0313\u0314\u0301\u0342\u0300\u0308\u0345]',
            re.UNICODE)
        self.vowels = re.compile('[αειηουωΑΕΙΗΟΥΩ]', re.UNICODE)
        self.grave = re.compile('\u0300', re.UNICODE)
        self.acute = re.compile('\u0301', re.UNICODE)
        self.sigma = re.compile('σ\b', re.UNICODE)
        self.sigma_alt = re.compile('ς', re.UNICODE)
        self.diacrit_sub1 = r'^{(' + \
                            self.diacriticals + \
                            r'}+)(' + \
                            self.vowels + \
                            r'}{2,})'
        self.diacrit_sub2 = r'^{(' + \
                            self.diacriticals + \
                            r'}+)(' + \
                            self.vowels + \
                            r'}{1})'

        self.lemmatizer = Lemmata('greek', 'lemmata')

    def normalize(token):
        """Normalize a single Greek word.

        Parameters
        ----------
        token : str
            The word to normalize.

        Returns
        -------
        normalized : str
            The normalized string.
        """
        # Remove non-alphabetic characters
        normalized = super(GreekTokenizer, self).normalize(token)

        # Convert grave accent to acute
        normalized = re.sub(self.grave, self.acute, token)

        # Remove diacriticals from vowels
        normalized = re.sub(self.diacrit_sub1, r'\2', normalized)
        normalized = re.sub(self.diacrit_sub2, r'\2\1', normalized)

        # Substitute sigmas
        normalized = re.sub(self.sigma, self.sigma_alt, normalized)
        return normalized

    def featurize(token):
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
        function.
        """
        # if isinstance
        features = {}
        features['lemmata'] = self.lemmatizer.lookup(token_type)[0][1]
        return features
