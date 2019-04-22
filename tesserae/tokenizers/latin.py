import collections
import re
import unicodedata

from cltk.semantics.latin.lookup import Lemmata
from cltk.stem.latin.j_v import JVReplacer

from tesserae.tokenizers.base import BaseTokenizer
from tesserae.db.entities import Frequency, Token


class LatinTokenizer(BaseTokenizer):
    def __init__(self, connection):
        super(LatinTokenizer, self).__init__(connection)

        # Set up patterns that will be reused
        self.jv_replacer = JVReplacer()
        self.lemmatizer = Lemmata('lemmata', 'latin')

        self.split_pattern = \
            '([<].+[>])| / | \. \. \.|\.\~\.\~\.|[^\w' + self.diacriticals + ']'

    # def tokenize(self, raw, record=True, text=None):
    #     normalized = unicodedata.normalize('NFKD', raw).lower()
    #     normalized = self.jv_replacer.replace(normalized)
    #     normalized = re.split(self.split_pattern, normalized, flags=re.UNICODE)
    #     display = re.split(self.split_pattern, raw, flags=re.UNICODE)
    #     featurized = self.featurize(normalized)
    #
    #     tokens = []
    #     frequencies = collections.Counter(
    #         [n for i, n in enumerate(normalized) if
    #          re.search('[\w]+', normalized[i], flags=re.UNICODE)])
    #     frequency_list = []
    #
    #     try:
    #         text_id = text.path
    #     except AttributeError:
    #         text_id = None
    #
    #     base = len(self.tokens)
    #
    #     for i, d in enumerate(display):
    #         idx = i + base
    #         if re.search('[\w]', d, flags=re.UNICODE):
    #             n = normalized[i]
    #             f = featurized[i]
    #             t = Token(text=text_id, index=idx, display=d, form=n, **f)
    #         else:
    #             t = Token(text=text_id, index=idx, display=d)
    #         tokens.append(t)
    #
    #     # Update the internal record if necessary
    #     if record:
    #         self.tokens.extend([t for t in tokens])
    #         self.frequencies.update(frequencies)
    #         frequencies = self.frequencies
    #         if '' in self.frequencies:
    #             del self.frequencies['']
    #
    #     print(frequencies)
    #     print(self.frequencies)
    #
    #     # Prep the freqeuncy objects
    #     for k, v in frequencies.items():
    #         f = Frequency(text=text_id, form=k, frequency=v)
    #         frequency_list.append(f)
    #
    #     return tokens, frequency_list

    def normalize(self, raw):
        """Normalize a Latin word.

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
        This function should be applied to Latin words prior to generating
        other features (e.g., lemmata).
        """
        # Apply the global normalizer
        normalized = super(LatinTokenizer, self).normalize(raw)

        # Replace j/v with i/u, respectively
        normalized = self.jv_replacer.replace(normalized)

        return normalized

    def featurize(self, tokens):
        """Lemmatize a Latin token.

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
        Input should be sanitized with `LatinTokenizer.normalize` prior to
        using this method.
        """
        if not isinstance(tokens, list):
            tokens = [tokens]
        lemmata = self.lemmatizer.lookup(tokens)
        features = []
        for i, l in enumerate(lemmata):
            features.append({'lemmata': [lem[0] for lem in l[1]]})
        return features
