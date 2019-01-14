import collections
import re
import unicodedata

from tesserae.db import FeatureSet, Frequency, Token


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
        self.word_characters = '[a-zA-Z]'
        self.diacriticals = \
            '\u0313\u0314\u0301\u0342\u0300\u0301\u0308\u0345'

        self.split_pattern = \
            '[<].+[>][\s]| / | \. \. \.|\.\~\.\~\.|[^\w' + self.diacriticals + ']'

        self.clear()

    def clear(self):
        """Reset the token list and frequency counter in this tokenizer."""
        self.tokens = []
        self.frequencies = collections.Counter()

    def featurize(self, tokens):
        raise NotImplementedError

    def normalize(self, raw):
        """Standardize token representation for further processing.

        The global version of this function removes whitespace, non-word
        characters, and digits from the lowercase form of each raw token.

        Parameters
        ----------
        raw : str or list of str
            The string(s) to convert. Whitespace will be removed to provide a
            list of tokens.

        Returns
        -------
        normalized : list
            The list of tokens in normalized form.
        """
        # If dealing with a list of strings, attempt to join the individual
        # string entries with spaces.
        if isinstance(raw, list):
            raw = ' '.join(raw)

        # Apply lowercase and NKFD normalization to the token string
        normalized = unicodedata.normalize('NFKD', raw).lower()
        #normalized = re.sub(r'[\n\r\r\n]', ' / ', normalized, flags=re.UNICODE)
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
        normalized = self.normalize(raw)
        normalized = re.split(self.split_pattern, normalized, flags=re.UNICODE)
        normalized = [n for n in normalized if n]
        raw = re.sub(r'<.+>\s', '', raw, flags=re.UNICODE)
        raw = re.sub(r'[\n]', r' / ', raw, flags=re.UNICODE)
        display = [t for t in re.split(r'( / )|([^\w])', raw, flags=re.UNICODE) if t]
        featurized = self.featurize(normalized)

        # Create the storage for this run of `tokenize`
        tokens = []
        frequencies = collections.Counter(
            [n for i, n in enumerate(normalized) if
             re.search('[\w]+', normalized[i], flags=re.UNICODE)])
        frequency_list = {}
        feature_sets = {}

        # Get the text id from the metadata if it was passed in
        try:
            text_id = text.id
        except AttributeError:
            text_id = None

        try:
            language = text.language
        except AttributeError:
            language = None

        # Prep the token objects
        base = len(self.tokens)
        norm_i = 0
        for i, d in enumerate(display):
            idx = i + base
            feature_set = None
            frequency = None
            if re.search(self.word_characters, d, flags=re.UNICODE):
                n = normalized[norm_i]
                f = featurized[norm_i]
                try:
                    feature_set = feature_sets[n]
                    frequency = freqeuncy_list[n]
                except KeyError as e:
                    feature_set = FeatureSet(
                        form=n, language=language,
                        frequency={str(text_id.id): frequencies[n]}, **f)
                    frequency = Frequency(text=text_id,
                                          form=n,
                                          frequency=frequencies[n])
                    frequency_list[n] = frequency
                norm_i += 1

            t = Token(text=text_id, index=idx, display=d,
                      feature_set=feature_set, frequency=frequency)
            tokens.append(t)

        # Update the internal record if necessary
        if record:
            if '' in self.frequencies:
                del self.frequencies['']
            self.tokens.extend([t for t in tokens])
            self.frequencies.update(frequencies)
            frequencies = self.frequencies

        return tokens, list(frequency_list.values())
