import collections
import re
import unicodedata

from tesserae.db import Frequency, Token


class BaseTokenizer(object):
    """Tokenizer with global operations.

    Attributes
    ----------
    tokens : list of tesserae.db.Token
        The tokens encountered by this tokenizer in the order they were
        encountered.
    frequencies : collections.Counter
        Key/value store of normalized token forms (key) and their reapective
        counts (value).

    Notes
    -----
    To create a tokenizer for a language not included in Tesserae, subclass
    BaseTokenizer and override the ``normalize`` and ``featurize`` methods with
    functionality specific to the new language.

    """

    def __init__(self):
        # This pattern is used over and over again
        self.diacriticals = \
            '\u0313\u0314\u0301\u0342\u0300\u0301\u0308\u0345'

        self.split_pattern = '(, )|([^\w' + self.diacriticals + '])'

        self.clear()

    def clear(self):
        """Reset the token list and frequency counter in this tokenizer."""
        self.tokens = []
        self.frequencies = collections.Counter()

    def featurize(self, tokens):
        raise NotImplementedError

    def normalize(self, tokens):
        """Standardize token representation for further processing.

        The global version of this function removes whitespace, non-word
        characters, and digits from the lowercase form of each raw token.

        Parameters
        ----------
        tokens : list of str
            The tokens to convert. Whitespace will be removed to provide a
            list of tokens.

        Returns
        -------
        normalized : list
            The list of tokens in normalized form.
        """
        if isinstance(tokens, str):
            tokens = [unicodedata.normalize('NFKD', t).lower() for t in
                      re.split('(, )|([^\w' + self.diacriticals + '])',
                               tokens, flags=re.UNICODE)
                      if t]
            print(tokens)
            # tokens =
            # tokens = re.sub(r'[\s—-]+', ' ', tokens.strip(),
            #                 flags=re.UNICODE)
            # tokens = re.sub(r'\'[s]{1}\s|\'[s]{1}$', ' s ', tokens,
            #                 flags=re.UNICODE)

        else:
            tokens = [unicodedata.normalize('NFKD', t).lower() for t in tokens]

        # normalized = \
        #     [re.sub(self.split_pattern, ' ', token.lower(),
        #             flags=re.UNICODE).strip()
        #      for token in tokens]

        normalized = [n for n in tokens if n != '']

        return normalized

    def tokenize(self, raw, record=True, text=None):
        """Normalize and featurize the words in a string.

        Tokens are comprised of the raw string, normalized form, and features
        related to the words under study. This computes all of the relevant
        data and tracks token frequencies in one shot.

        Parameters
        ----------
        raw : str or list of str
            The string(s) to process. If a list, assumes that the string
            members of the list have already been split as intended (e.g.
            list elements were split on whitespace).
        record : bool
            If True, record tokens and frequencies in `self.tokens` and
            `self.frequencies`, respectively. Pass False to prevent recording
            in the event that the string is re-processed.
        text : tesserae.Text, optional
            Text metadata for associating tokens and frequencies with a
            particular text.

        Returns
        -------
        tokens : list of tesserae.db.Token
        frequencies : list of tesserae.db.Frequency
        """
        # Get the correct attributes set up if they have not been.
        if not hasattr(self, 'tokens') or not hasattr(self, 'frequencies'):
            self.clear()

        # Compute the display, normalized, and featurized forms of the tokens
        if isinstance(raw, str):
            #raw = re.sub(r'\s+', ' ', raw)
            display = [s for s in re.split(self.split_pattern, raw,
                                           flags=re.UNICODE)
                       if s]
            if display[0] == ' ':
                display = display[1:]
            if display[-1] == ' ':
                display = display[:-1]
        else:
            display = raw
        # print(display)
        normalized = self.normalize(display)
        # print(normalized)
        featurized = self.featurize(normalized)

        tokens = []
        frequencies = collections.Counter(normalized)
        frequency_list = []

        try:
            text_id = text.id
        except AttributeError:
            text_id = None

        # Prep the token objects
        for i, d in enumerate(display):
            if re.search(r'^\w+$', d, flags=re.UNICODE):
                n = normalized[i]
                f = featurized[i]
                t = Token(text=text_id, index=i, display=d, form=n, **f)
            else:
                t = Token(text=text_id, index=i, display=d)
            tokens.append(t)

        # Update the internal record if necessary
        if record:
            self.tokens.extend([t for t in tokens])
            self.frequencies.update(frequencies)
            frequencies = self.frequencies

        # Prep the freqeuncy objects
        for k, v in frequencies.items():
            f = Frequency(text=text_id, form=k, frequency=v)
            frequency_list.append(f)

        return tokens, frequencies
