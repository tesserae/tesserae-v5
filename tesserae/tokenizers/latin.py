import re

from cltk.semantics.latin.lookup import Lemmata
from cltk.stem.latin.j_v import JVReplacer

from tesserae.tokenizers.base import BaseTokenizer
from tesserae.features.trigrams import tri_latin


class LatinTokenizer(BaseTokenizer):
    def __init__(self, connection):
        super(LatinTokenizer, self).__init__(connection)

        # Set up patterns that will be reused
        self.jv_replacer = JVReplacer()
        self.lemmatizer = Lemmata('lemmata', 'lat')

        self.split_pattern = \
            '( / )|([\\s]+)|([^\\w' + self.diacriticals + ']+)'

    def normalize(self, raw, split=True):
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
        normalized, tags = super(LatinTokenizer, self).normalize(raw)

        # Replace j/v with i/u, respectively
        normalized = self.jv_replacer.replace(normalized)

        if split:
            normalized = re.split(self.split_pattern,
                                  normalized,
                                  flags=re.UNICODE)
            normalized = [
                t for t in normalized if t and re.search(r'[\w]+', t)
            ]

        return normalized, tags

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
        fixed_lemmata = []
        for lemma in lemmata:
            lem_lemmata = [lem[0] for lem in lemma[1]]
            fixed_lemmata.append(lem_lemmata)

        grams = tri_latin(tokens)
        features = {'lemmata': fixed_lemmata, 'sound': grams}
        return features
