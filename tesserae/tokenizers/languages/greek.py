import re
import unicodedata

from cltk.semantics.latin.lookup import Lemmata

from tesserae.tokenizers.languages.base import BaseTokenizer


class GreekTokenizer(BaseTokenizer):
    def __init__(self):
        super(GreekTokenizer, self).__init__()

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

    def normalize(self, tokens):
        """Normalize a single Greek word.

        Parameters
        ----------
        token : list of str
            The word to normalize.

        Returns
        -------
        normalized : str
            The normalized string.
        """
        if isinstance(tokens, str):
            tokens = \
                [t for t in re.split('([^\w' + self.diacriticals + '])',
                                     tokens,
                                     flags=re.UNICODE)
                 if t]
        # Normalize prior to processing
        normalized = [unicodedata.normalize('NFKD', t).lower() for t in tokens]
        # normalized = normalized.lower()

        # Convert grave accent to acute
        normalized = \
            [re.sub(self.grave, self.acute, n, flags=re.UNICODE)
             for n in normalized]

        # Remove diacriticals from vowels
        normalized = \
            [re.sub(self.diacrit_sub1, r' \2', n, flags=re.UNICODE)
             for n in normalized]
        normalized = \
            [re.sub(self.diacrit_sub2, r' \2\1', n, flags=re.UNICODE)
             for n in normalized]

        # Special case for some capitals with diacriticals
        # normalized = re.sub('\u0313α', 'α\u0313', normalized, flags=re.UNICODE)
        # normalized = re.sub('\u0314η', 'η\u0314', normalized, flags=re.UNICODE)

        # Substitute sigmas
        normalized = \
            [re.sub(self.sigma, self.sigma_alt, n, flags=re.UNICODE)
             for n in normalized]

        # Remove punctuation
        # normalized = re.sub('([,.!?:;])(\w)', '\1 \2',
        #                     normalized, flags=re.UNICODE)
        normalized = \
            [re.sub('[,.!?;:\'"\(\)†\d\u201C\u201D—-]+', ' ',
                    n, flags=re.UNICODE)
             for n in normalized]

        normalized = \
            [re.split('[\s]+', n.strip(), flags=re.UNICODE)
             for n in normalized]

        return normalized

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
        features = []
        lemmata = self.lemmatizer.lookup(token_type)
        for i, l in enumerate(lemmata):
            features.append({'lemmata': lemmata[i][1]})
        return features

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
        display = \
            [t for t in re.split('([^\w' + self.diacriticals + '])', raw,
                                 flags=re.UNICODE)
             if t]

        return \
            super(GreekTokenizer, self).tokenize(
                display, record=record, text=text)
