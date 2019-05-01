import re

from tesserae.db import Text, Token, Unit
from tesserae.tokenizers import BaseTokenizer


class InvalidMetadataError(Exception):
    """Raised when a bad text metadata object is supplied."""
    def __init__(self, metadata):
        msg = 'Text metadata must be an instance of tesserae.db.Text. Bad '
        msg += 'metadata: {}'
        super(InvalidMetadataError, self).__init__(metadata.__class__.__name__)


class InvalidTokenError(Exception):
    """Raised when attempting to unitize an object not tesserae.db.Token."""
    def __init__(self, token):
        msg = 'Attempted to create a unit from an object that was not an '
        msg += 'instance of tesserae.db.Token. Preprocess tokens with a '
        msg += 'Tesserae tokenizer before unitizing. Bad token: {}'
        super(InvalidTokenError, self).__init__(
            msg.format(token.__class__.__name__))


class InvalidTokenizerError(Exception):
    """Raised when attempting to unitize a string but bad tokenizer supplied"""
    def __init__(self, tokenizer):
        msg = 'A string was supplied to the Unitizer but an invalid tokenizer '
        msg += 'was supplied to process the string. Please supply a tokenizer '
        msg += 'from tesserae.tokenizers to continue. Bad tokenizer: {}'
        super(InvalidTokenizerError, self).__init__(
            msg.format(tokenizer.__class__.__name__))


class Unitizer(object):
    """Group tokens into units.

    Attributes
    ----------
    lines : list of tesserae.db.Unit
        Line units created from supplied tokens.
    phrases : list of tesserae.db.Unit
        Phrase units created from supplied tokens.

    See Also
    --------
    tesserae.tokenizers
    """
    def __init__(self):
        self.clear()

    def clear(self):
        """Reset this unitizer to contain no units."""
        self.lines = []
        self.phrases = []

    def unitize(self, tokens, tags, metadata, tokenizer=None, stop=False):
        """Split a poem into line and phrase units.

        Parameters
        ----------
        tokens : str or list of tesserae.db.Tokens
            The tokens to group as units.
        metadata : tesserae.db.Text
            Text metadata for assigning the units to a text.
        tokenizer : tesserae.tokenizers.languages.BaseTokenizer

        Notes
        -----
        This method requires preprocessing the tokens with one of the
        tesserae tokenizers.

        See Also
        --------
        tesserae.tokenizers
        """
        # If a string was passed in, run it through a tokenizer
        if isinstance(tokens, str):
            # Make sure that the tokenizer is a valid tokenizer
            if not isinstance(tokenizer, BaseTokenizer):
                raise InvalidTokenizerError(tokenizer)
            # Get the tokens and ignore the frequencies
            tokens, _ = tokenizer.tokenize(tokens)

        # Check that the metadata object is valid text metadata
        if not isinstance(metadata, Text):
            raise InvalidMetadataError(metadata)

        # Create initial line/phrase objects
        if len(self.lines) == 0:
            self.lines.append(
                Unit(text=metadata,
                     index=len(self.lines),
                     unit_type='line'))
        if len(self.phrases) == 0:
            self.phrases.append(
                Unit(text=metadata,
                     index=len(self.phrases),
                     unit_type='phrase'))

        tag_idx = 0
        try:
            if tags[tag_idx] not in self.lines[-1].tags:
                self.lines[-1].tags.append(tags[tag_idx])
            if tags[tag_idx] not in self.phrases[-1].tags:
                self.phrases[-1].tags.append(tags[tag_idx])
        except IndexError:
            pass

        # Add the token to the current line and phrase and determine if it is
        # a unit delimiter.
        for i, t in enumerate(tokens):
            # Ensure that the token is valid
            if not isinstance(t, Token):
                raise InvalidTokenError(t)

            # Search for a phrase delimiter
            phrase_delim = re.search(r'[.?!;:]', t.display, flags=re.UNICODE)
            word = re.search(r'[\w]', t.display, flags=re.UNICODE)

            # Get the current line and phrase
            # if '<' not in t.display:
            self.lines[-1].tokens.append(t)
            t.line = self.lines[-1]

            # Handle seeing multiple phrase delimiters in a row
            if len(self.phrases) > 1 and not word and len(self.phrases[-1].tokens) == 0:
                self.phrases[-2].tokens.append(t)
                t.phrase = self.phrases[-2]
            else:
                self.phrases[-1].tokens.append(t)
                t.phrase = self.phrases[-1]

            # If this token contains a phrasee delimiter (one of .?!;:),
            # create a new phrase unit and append it for the next iteration.
            if phrase_delim and len(self.phrases[-1].tokens) > 0:
                self.phrases.append(
                    Unit(text=metadata,
                         index=len(self.phrases),
                         unit_type='phrase'))

                try:
                    if tags[tag_idx] not in self.phrases[-1].tags:
                        self.phrases[-1].tags.append(tags[tag_idx])
                except IndexError:
                    pass

            # If this token contains a newline or the Tesserae line delimiter,
            # create a new line unit and append it for the next iteration.
            if re.search(r'([\n])|( / )', t.display, flags=re.UNICODE) and len(self.lines[-1].tokens) > 1:
                if len(self.phrases[-1].tokens) == 0:
                    self.phrases[-1].tags.pop()

                self.lines.append(
                    Unit(text=metadata,
                         index=len(self.lines),
                         unit_type='line'))
                tag_idx += 1

                try:
                    if tags[tag_idx] not in self.lines[-1].tags:
                        self.lines[-1].tags.append(tags[tag_idx])
                    if tags[tag_idx] not in self.phrases[-1].tags:
                        self.phrases[-1].tags.append(tags[tag_idx])
                except IndexError:
                    pass

        if stop and len(self.lines[-1].tokens) == 0:
            self.lines.pop()

        if stop:
            for i in range(len(self.phrases) - 1, 0, -1):
                if len(self.phrases[i].tokens) < 2:
                    self.phrases.pop()
                else:
                    break

        return self.lines, self.phrases
