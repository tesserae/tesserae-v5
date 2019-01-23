"""Database standardization for token features.

Classes
-------
FeatureSet
    Token features data model.
"""
import typing

from tesserae.db.entities import Entity


class FeatureSet(Entity):
    """Standard features derived for tokens.

    Tokens contain the atomic pieces of text that inform Matches and make
    up Units. Tesserae computes the following features for each token: the
    normalized form of the token, lemmata and semantic meaning, and sound
    trigrams.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    form : str, optional
        The normalized form of this token. Normalization depends of the
        language of the token, however this typically includes converting
        characters to lower case and standardizing diacritical marks.
    lemmata : list of str, optional
        List of stem word associated with this token.
    semantic : list of str, optional
    sound : list of str, optional
    language : str, optional
        The language that this token feature set belongs to.

    Attributes
    ----------
    id : bson.objectid.ObjectId
        Database id of the text. Should not be set locally.
    form : str
        The normalized form of this token. Normalization depends of the
        language of the token, however this typically includes converting
        characters to lower case and standardizing diacritical marks.
    lemmata : list of str
        List of stem word associated with this token.
    semantic : list of str
    sound : list of str
    language : str
        The language that this token feature set belongs to.
    """

    collection = 'feature_sets'

    def __init__(self, id=None, form=None, lemmata=None, semantic=None,
                 sound=None, language=None, frequency=None):
        super(FeatureSet, self).__init__(id=id)
        self.form: typing.Optional[str] = form
        self.lemmata: typing.List[str] = lemmata if lemmata is not None else []
        self.semantic: typing.List[str] = \
            semantic if semantic is not None else []
        self.sound: typing.List[str] = sound if sound is not None else []
        self.language: typing.Optional[str] = language

    def __hash__(self):
        return hash(self.form)

    def __str__(self):
        return '{' + self.form + '}'

    def unique_values(self):
        uniques = {'form': self.form}
        return uniques
