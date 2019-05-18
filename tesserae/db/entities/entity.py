"""Base entity for database ease of access.

Classes
-------
Entity
    Base class for other entities.

"""
import copy
import typing

from bson.objectid import ObjectId


class Entity():
    """Abstract database entity.

    This object represents a generic database entry (e.g., a document in
    MongoDB).
    """

    collection = None

    def __init__(self, id=None):
        self._id: typing.Optional[ObjectId] = id

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        vals = self.unique_values()
        return hash(str(vals)) if not isinstance(vals, dict) \
            else hash(' '.join([str(vals[key]) for key in sorted(list(vals.keys()))]))

    def copy(self):
        """Create a deep copy of this Entity.

        Returns
        -------
        entity : tesserae.db.entities.Entity
            A copy of this entity.
        """
        attrs = self.json_encode()
        if '_id' in attrs:
            attrs['id'] = attrs['_id']
            del attrs['_id']
        return self.__class__(**attrs)

    @property
    def id(self) -> typing.Optional[ObjectId]:
        """The database id of this entity.

        Returns
        -------
        id : str or bson.ObjectId or None
        """
        return self._id

    @id.setter
    def id(self, value: typing.Optional[ObjectId]):
        self._id = value

    def json_encode(self, exclude: typing.Optional[typing.List[str]] = None):
        """Encode this entity as a valid JSON object.

        Parameters
        ----------
        exclude : list of str, optional
            List of attributes to exclude from the result.

        Raises
        ------
        AttributeError
            Raised when a non-existent attribute is encountered in ``exclude``.
        """
        exclude = exclude if exclude is not None else []
        exclude = exclude + ['_ignore']

        obj = {k: v for k, v in self.__dict__.items() if k not in exclude}
        if '_id' in obj:
            obj_id = obj['_id']
            del obj['_id']
            obj['id'] = str(obj_id)

        return obj

    @classmethod
    def json_decode(cls, obj: typing.Dict[str, typing.Any]):
        """Decode a JSON object to create an entity.

        Parameters
        ----------
        obj : dict
            Dictionary with key/value pairs corresponding to Entity object
            attributes.

        Returns
        -------
        entity : `tesserae.db.entities.Entity`
            Entity created from ``obj``.

        Raises
        ------
        bson.errors.InvalidId
            Raised when obj['_id'] has a value incompatible with ObjectId
        TypeError
            Raised when obj['_id'] has a type incompatible with ObjectId
        """
        if '_id' in obj:
            obj['id'] = ObjectId(obj['_id'])
            del obj['_id']
        instance = cls()
        for k, v in obj.items():
            setattr(instance, k, v)
        return instance

    def unique_values(self):
        return {'id': self.id}
