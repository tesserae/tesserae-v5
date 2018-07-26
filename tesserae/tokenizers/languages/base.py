import collections
import re


class BaseTokenizer(object):
    """Tokenizer with global operations.

    Notes
    -----
    To create a tokenizer for a language not included in Tesserae, subclass
    BaseTokenizer and override the ``normalize`` and ``featurize`` methods with
    functionality specific to the new language.

    """
    def __init__(self):
        self.tokens = []
        self.frequencies = collections.Counter()

    def __call__(self, tokens):
        normalized = self.tokens

    def normalize(self, tokens):
        """Standardize token representation for further processing.

        The global version of this function removes whitespace, non-word
        characters, and digits from the lowercase form of each raw token.

        Parameters
        ----------
        tokens : str
            The tokens to convert. Whitespace will be removed to provide a
            list of tokens.

        Returns
        -------
        normalized : list
            The list of tokens in normalized form.
        """
        if isinstance(tokens, str):
            tokens = re.sub(r'\s', ' ', tokens.strip(), flags=re.UNICODE)
            tokens = tokens.split(' ')

        normalized = [re.sub(r'\W', '', token.lower(), flags=re.UNICODE)
                      for token in tokens]
        return normalized

    def featurize(self, tokens):
        raise NotImplementedError
