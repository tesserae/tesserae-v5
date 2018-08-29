import re
import unicodedata

from cltk.semantics.latin.lookup import Lemmata
from cltk.stem.latin.j_v import JVReplacer

from tesserae.tokenizers.languages.base import BaseTokenizer


class LatinTokenizer(BaseTokenizer):
    def __init__(self):
        super(LatinTokenizer, self).__init__()

        # Set up patterns that will be reused
        self.jv_replacer = JVReplacer()
        self.lemmatizer = Lemmata('lemmata', 'latin')

    def normalize(self, tokens):
        """Normalize a Latin word.

        Parameters
        ----------
        tokens : str or list of str
            The word(s) to normalize.

        Returns
        -------
        normalized : list of str
            The normalized string.

        Notes
        -----
        This function should be applied to Latin words prior to generating
        other features (e.g., lemmata).
        """
        if isinstance(tokens, str):
            unicodedata.normalize('NFKD', tokens)
            normalized = self.jv_replacer.replace(tokens.lower())
            normalized = re.split(self.split_pattern, normalized, flags=re.UNICODE)
        else:
            normalized = [self.jv_replacer.replace(t.lower()) for t in tokens]

        # Run through the remaining normalization and return.
        return [n for n in normalized if n]

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
        lemmata = self.lemmatizer.lookup(tokens)
        features = []
        for i, l in enumerate(lemmata):
            features.append({'lemmata': lemmata[i][1]})
        return features
