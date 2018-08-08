import re
import unicodedata

from cltk.semantics.latin.lookup import Lemmata

from tesserae.tokenizers.languages.base import BaseTokenizer


class GreekTokenizer(BaseTokenizer):
    def __init__(self):
        # Set up patterns that will be reused
        self.diacriticals = \
            '[\u0313\u0314\u0301\u0342\u0300\u0301\u0308\u0345]'
        self.vowels = '[αειηουωΑΕΙΗΟΥΩ]'
        self.grave = '\u0300'
        self.acute = '\u0301'
        self.sigma = 'σ\b'
        self.sigma_alt = 'ς'
        self.diacrit_sub1 = \
            '\s(' + self.diacriticals + '+)(' + self.vowels + '{2,})'
        self.diacrit_sub2 = \
            '\s(' + self.diacriticals + '+)(' + self.vowels + '{1})'

        self.lemmatizer = Lemmata('lemmata', 'greek')

    def normalize(self, token):
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
        # Normalize prior to processing
        normalized = unicodedata.normalize('NFKD', token)
        normalized = normalized.lower()

        # Convert grave accent to acute
        normalized = re.sub(self.grave, self.acute,
                            normalized, flags=re.UNICODE)

        # Remove diacriticals from vowels
        normalized = re.sub(self.diacrit_sub1, r' \2',
                            normalized, flags=re.UNICODE)
        normalized = re.sub(self.diacrit_sub2, r' \2\1',
                            normalized, flags=re.UNICODE)

        # Special case for some capitals with diacriticals
        # normalized = re.sub('\u0313α', 'α\u0313', normalized, flags=re.UNICODE)
        # normalized = re.sub('\u0314η', 'η\u0314', normalized, flags=re.UNICODE)

        # Substitute sigmas
        normalized = re.sub(self.sigma, self.sigma_alt,
                            normalized, flags=re.UNICODE)

        # Remove punctuation
        # normalized = re.sub('([,.!?:;])(\w)', '\1 \2',
        #                     normalized, flags=re.UNICODE)
        normalized = re.sub('[,.!?;:\'"\(\)†\d\u201C\u201D—-]+', ' ',
                            normalized, flags=re.UNICODE)

        normalized = re.split('[\s]+', normalized.strip(), flags=re.UNICODE)

        return [n for n in normalized if n != '']

    def featurize(self, token):
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
