"""Database standardization for feature instances

Classes
-------
Property
    Feature instance data model, located by unit and position.
"""
import typing

from bson.objectid import ObjectId

from tesserae.db.entities.entity import Entity
from tesserae.db.entities.text import Text
from tesserae.db.entities.unit import Unit


class Property(Entity):
    """An instance of a feature at a given position of a particular unit

    Parameters/Attributes
    ---------------------
    id : bson.ObjectId, optional
        Database id of the text. Should not be set locally.
    feature_type : str
        Feature type this Property represents
    feature_index : int
        Index of Feature type this Property represents
    language : str
        Language of the Feature type this Property represents
    unit : bson.ObjectId
        Unit to which this Property belongs
    unit_index : int
        Index of the Unit in which this Property is found
    unit_type : str
        How the chunk of text in the Unit to which this Property belongs was
        defined, e.g., "line", "phrase", etc.
    position : int
        Index of position which this Property occupies in the context of its
        Unit
    text : bson.ObjectId
        Text of the Unit to which this Property belongs

    """

    collection = 'properties'

    def __init__(self, id=None, feature_type=None, feature_index=None,
                 language=None, unit=None, unit_index=None, unit_type=None,
                 position=None, text=None):
        super(Property, self).__init__(id=id)
        self.feature_type: str = feature_type
        self.feature_index: int = feature_index
        self.language: str = language
        self.unit: typing.Union[ObjectId, Unit] = unit
        self.unit_index: int = unit_index
        self.unit_type: str = unit_type
        self.position: int = position
        self.text: typing.Union[ObjectId, Text] = text

    def json_encode(self, exclude=None):
        self._ignore = [self.unit, self.text]
        if isinstance(self.unit, Entity):
            self.unit = self.unit.id
        if isinstance(self.text, Entity):
            self.text = self.text.id

        obj = super(Unit, self).json_encode(exclude=exclude)

        self.unit, self.text = self._ignore
        del self._ignore

        return obj

    def unique_values(self):
        uniques = {
            'feature_type': self.feature_type,
            'feature_index': self.feature_index,
            'language': self.language,
            'unit':
                self.unit.id if isinstance(self.unit, Entity)
                else self.unit,
            'position': self.position,
        }
        return uniques

    def __repr__(self):
        return (
            f'Property(feature_type={self.feature_type}, '
            f'feature_index={self.feature_index}, language={self.language}, '
            f'unit={self.unit}, unit_index={self.unit_index}, '
            f'unit_type={self.unit_type}, position={self.position}, '
            f'text={self.text})'
        )
