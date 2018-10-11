"""Database standardization for text tokens.

Classes
-------
Token
    Text token data model with matching-related features.
"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities import Entity


class Token(Entity):
    """An atomic piece of text, along with related features.

    Tokens contain the atomic pieces of text that inform Matches and make
    up Units. In addition to the raw text, the normalized form of the text
    and features like lemmata and semantic meaning are also part of a
    token.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    text : str or bson.objectid.ObjectId, optional
        The text containing this token.
    index : int, optional
        The order of this token in the text.
    display : str, optional
        The un-altered form of this token, as it appears in the original
        text.
    form : str, optional
        The normalized form of this token. Normalization depends of the
        language of the token, however this typically includes converting
        characters to lower case and standardizing diacritical marks.
    lemmata : list of str, optional
        List of stem word associated with this token.
    semantic : list of str, optional
    sound : list of str, optional

    Attributes
    ----------
    id : bson.objectid.ObjectId
        Database id of the text. Should not be set locally.
    text : str or bson.objectid.ObjectId
        The text containing this token.
    index : int
        The order of this token in the text.
    display : str
        The un-altered form of this token, as it appears in the original
        text.
    form : str
        The normalized form of this token. Normalization depends of the
        language of the token, however this typically includes converting
        characters to lower case and standardizing diacritical marks.
    lemmata : list of str
        List of stem word associated with this token.
    semantic : list of str
    sound : list of str

    """

    collection = 'tokens'

    def __init__(self, id=None, text=None, index=None, display=None, form=None,
                 lemmata=None, semantic=None, sound=None):
        super(Token, self).__init__(id=id)
        self.text: typing.Optional[typing.Union[str, ObjectId]] = text
        self.index: typing.Optional[int] = index
        self.display: typing.Optional[str] = display
        self.form: typing.Optional[str] = form
        self.lemmata: typing.List[str] = lemmata if lemmata is not None else []
        self.semantic: typing.List[str] = \
            semantic if semantic is not None else []
        self.sound: typing.List[str] = sound if sound is not None else []

    def match(other, feature):
        """Determine whether two tokens match along a given feature.

        Parameters
        ----------
        other : tesserae.db.entities.Token
            The token to compare against.
        feature : {'form','lemmata','semantic','lemmata + semantic','sound'}
            The feature to compare on.

        Returns
        -------
        match : bool
        """
        if feature != 'lemmata + semantic':
            return getattr(self, feature) == getattr(other, feature)
        else:
            return self.lemmata == other.lemmata and \
                   self.semantic == other.semantic
