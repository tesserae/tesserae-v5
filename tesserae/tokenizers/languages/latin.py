import re

from cltk.semantics.latin.lookup import Lemmata
from cltk.stem.latin.j_v import JVReplacer

from tesserae.tokenizers.languages.base import BaseTokenizer


class LatinTokenizer(BaseTokenizer):
    def __init__(self):
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
        if isinstance(tokens, list):
            tokens = ' '.join(tokens)
        normalized = self.jv_replacer.replace(tokens.lower())
        return super(LatinTokenizer, self).normalize(normalized)

    def featurize(self, token):
        """Lemmatize a Latin token.

        Parameters
        ----------
        token : str
            The token to featurize.

        Returns
        -------
        lemmata : dict
            The features for the token.

        Notes
        -----
        Input should be sanitized with `latin_normalizer` prior to using this
        function.
        """
        lemmata = self.lemmatizer.lookup(token)[0][1]
        return lemmata
