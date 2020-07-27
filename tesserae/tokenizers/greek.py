import re

from cltk.semantics.latin.lookup import Lemmata

from tesserae.tokenizers.base import BaseTokenizer
from tesserae.features.trigrams import trigrammify


class GreekTokenizer(BaseTokenizer):
    def __init__(self, connection):
        super(GreekTokenizer, self).__init__(connection)

        # Set up patterns that will be reused
        self.vowels = 'αειηουωΑΕΙΗΟΥΩ'
        self.grave = '\u0300'
        self.acute = '\u0301'
        self.sigma = 'σ\b'
        self.sigma_alt = 'ς'
        # diacriticals should not be considered part of ``word_characters`` so
        # that extraneous diacritical marks unattended by a proper word
        # character to bind to do not appear as proper words during
        # tokenization of display tokens (see BaseTokenizer.tokenize);
        # also ignore the middle dot character, which is a punctuation mark
        self.word_regex = re.compile('[ΆΈ-ώ' + self.sigma_alt + ']+',
                                     flags=re.UNICODE)

        self.diacrit_sub1 = r'[\s.,;?!]([' + \
            self.diacriticals + ']+)([' + self.vowels + ']{2,})'
        self.diacrit_sub2 = r'[\s.,;?!]([' + \
            self.diacriticals + ']+)([' + self.vowels + ']{1})'

        self.split_pattern = ''.join([
            '( / )|([\\s]+)|([^\\w\\d', self.diacriticals, self.sigma_alt,
            r"])"
        ])

        self.lemmatizer = Lemmata('lemmata', 'grc')

    def normalize(self, raw, split=True):
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
        normalized, tags = super(GreekTokenizer, self).normalize(raw)

        # Convert grave accent to acute
        normalized = re.sub(self.grave,
                            self.acute,
                            normalized,
                            flags=re.UNICODE)

        # Remove diacriticals from vowels
        normalized = re.sub(self.diacrit_sub1,
                            r' \2',
                            normalized,
                            flags=re.UNICODE)
        normalized = re.sub(self.diacrit_sub2,
                            r' \2\1',
                            normalized,
                            flags=re.UNICODE)

        # Substitute sigmas
        normalized = re.sub(self.sigma,
                            self.sigma_alt,
                            normalized,
                            flags=re.UNICODE)

        # Remove digits and single-quotes from the normalized output
        normalized = re.sub(r"['\d]+", r' ', normalized, flags=re.UNICODE)

        # Split the output into a list of normalized tokens if requested
        if split:
            normalized = re.split(self.split_pattern,
                                  normalized,
                                  flags=re.UNICODE)
            normalized = [
                t for t in normalized if t and re.search(r'[\w]+', t)
            ]

        return normalized, tags

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
        lemmata = self.lemmatizer.lookup(tokens)
        fixed_lemmata = []
        for lemma in lemmata:
            lem_lemmata = [lem[0] for lem in lemma[1]]
            fixed_lemmata.append(lem_lemmata)

        grams = trigrammify(tokens)
        features = {'lemmata': fixed_lemmata, 'sound': grams}
        return features
