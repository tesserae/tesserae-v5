import collections
import re
import unicodedata

from cltk.semantics.latin.lookup import Lemmata
from cltk.stem.latin.j_v import JVReplacer

from tesserae.tokenizers.base import BaseTokenizer
from tesserae.db.entities import Token


class LatinTokenizer(BaseTokenizer):
    def __init__(self, connection):
        super(LatinTokenizer, self).__init__(connection)

        # Set up patterns that will be reused
        self.jv_replacer = JVReplacer()
        self.lemmatizer = Lemmata('lemmata', 'latin')

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
            normalized = re.split(self.split_pattern, normalized, flags=re.UNICODE)
            normalized = [t for t in normalized
                          if t and self.word_regex.search(t)]

        return normalized, tags

    def featurize(self, tokens):
        """Lemmatize Latin tokens.

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
        >>> tokens = ['arma', 'cano']

        Then the result would look something like this:
        >>> result = {
        >>>     'lemmata': [
        >>>         ['arma', 'armo'],
        >>>         ['canus', 'cano']
        >>>     ]
        >>> }

        Note that `result['lemmata'][0]` is a list containing the lemmata for
        `tokens[0]`; similarly, `result['lemmata'][1]` is a list containing the
        lemmata for `tokens[1]`.

        Notes
        -----
        Input should be sanitized with `LatinTokenizer.normalize` prior to
        using this method.

        """
        if not isinstance(tokens, list):
            tokens = [tokens]
        lemmata = self.lemmatizer.lookup(tokens)
        fixed_lemmata = []
        for lem in lemmata:
            lem_lemmata = [l[0] for l in lem[1]]
            fixed_lemmata.append(lem_lemmata)
        features = {
            'lemmata': fixed_lemmata
        }
        return features
