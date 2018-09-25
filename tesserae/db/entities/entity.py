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
        self._id: typing.Optional[typing.Union[str, ObjectId]] = id

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

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
    def id(self) -> typing.Optional[typing.Union[str, ObjectId]]:
        """The database id of this entity.

        Returns
        -------
        id : str or bson.ObjectId or None
        """
        return self._id

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
        obj = copy.deepcopy(self.__dict__)
        exclude = exclude if exclude is not None else []
        for k in exclude:
            if k in obj:
                del obj[k]
        return obj

    @classmethod
    def json_decode(cls, obj: typing.Dict[str, typing.Any]):
        """Decode a JSON object to create an entity.

        Parameters
        ----------
        obj : dict
            Dictionary with key/value pairs corersponding to Entity object
            attributes.

        Returns
        -------
        entity : `tesserae.db.entities.Entity`
            Entity created from ``obj``.

        Raises
        ------
        AttributeError
            If ``obj`` contains
        """
        if '_id' in obj:
            obj['id'] = obj['_id']
            del obj['_id']
        return cls(**obj)
