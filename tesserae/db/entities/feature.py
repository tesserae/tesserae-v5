"""Database standardization for token features.

Classes
-------
Feature
    Data model for a token feature (e.g., form, lemma, semantic, 3gr).
"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities.entity import Entity


class Feature(Entity):
    """Data model for a token feature (e.g., form, lemma, semantic, 3gr).

    Features are processed versions of tokens that have meaning for intertext
    searches. Possible features include the normalized forms of tokens, stem
    words, semantic meaning, etc. Each individual feature is granted a unique
    index, which is used to encode the set of all features of a particular
    type.

    Parameters
    ----------
    id : ObjectId, optional
        Database id of the Feature.
    feature : {'form','lemma','semantic','sound'}, optional
        The type of feature.
    token : str, optional
        The string representation of the feature.
    frequencies : dict
        Mapping of frequency data per text.
    semantic : ObjectId or tesserae.db.entities.Feature
        Semantic data tied to the form or lemma.
    sound : ObjectId or tesserae.db.entities.Feature
        Sound trigram data tied to the form or lemma.
    """

    def __init__(self, id=None, feature=None, token=None, index=None,
                 frequencies=None, semantic=None, sound=None):
        super(Feature, self).__init__(id=id)
        self.feature: typing.Optional[str] = feature
        self.token: typing.Optional[str] = token
        self.index: typing.Optional[int] = index
        self.frequencies: typing.Dict[ObjectId, int] = \
            frequencies if frequencies is not None else {}

    def json_encode(self):
        self._ignore = [self.semantic, self.sound]

        if isinstance(self.semantic, Entity):
            self.semantic = self.semantic.id
        if isinstance(self.sound, Entity):
            self.sound = self.sound.id

        obj = super(Feature, self).json_encode()

        self.semantic, self.sound = self._ignore
        del self._ignore

        return obj
