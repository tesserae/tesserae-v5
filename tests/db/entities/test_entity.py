import pytest

from bson.objectid import ObjectId

from tesserae.db.entities.entity import Entity


@pytest.fixture(scope='module')
def valid_ids():
    return [None, 'foo', ObjectId('0123456789ab0123456789ab')]


class TestEntity(object):
    __entity_class__ = Entity

    def test_init(self, valid_ids):
        e = self.__entity_class__()
        assert hasattr(e, '_id')
        assert e._id is None

        for v in valid_ids:
            e = self.__entity_class__(id=v)
            assert hasattr(e, '_id')
            assert e._id is v

    def test_eq(self, valid_ids):
        for i in range(len(valid_ids)):
            for j in range(len(valid_ids)):
                e1 = self.__entity_class__(id=valid_ids[i])
                e2 = self.__entity_class__(id=valid_ids[j])

                if i != j:
                    assert e1 != e2
                else:
                    assert e1 == e2

    def test_copy(self, valid_ids):
        e1 = self.__entity_class__()
        e2 = e1.copy()
        assert e2 is not e1
        assert e2 == e1

        for v in valid_ids:
            e1 = self.__entity_class__(id=v)
            e2 = e1.copy()
            assert e2 is not e1
            assert e2 == e1

    def test_id(self, valid_ids):
        e = self.__entity_class__()
        assert e.id == e._id

        for v in valid_ids:
            e = self.__entity_class__(id=v)
            assert e.id == e._id

    def test_json_encode(self, valid_ids):
        e = self.__entity_class__()
        assert e.json_encode() == {'_id': None}

        e = self.__entity_class__()
        assert e.json_encode(exclude=['foo']) == {'_id': None}

        e = self.__entity_class__()
        assert e.json_encode(exclude=['_id']) == {}

        e = self.__entity_class__()
        assert e.json_encode(exclude=['_id', 'foo']) == {}

        e = self.__entity_class__()
        assert e.json_encode(exclude=['foo', '_id']) == {}

        for v in valid_ids:
            e = self.__entity_class__(id=v)
            assert e.json_encode() == {'_id': v}

            e = self.__entity_class__(id=v)
            assert e.json_encode(exclude=['foo']) == {'_id': v}

            e = self.__entity_class__(id=v)
            assert e.json_encode(exclude=['_id']) == {}

            e = self.__entity_class__(id=v)
            assert e.json_encode(exclude=['_id', 'foo']) == {}

            e = self.__entity_class__(id=v)
            assert e.json_encode(exclude=['foo', '_id']) == {}

    def test_json_decode(self, valid_ids):
        e = self.__entity_class__()
        decoded = self.__entity_class__.json_decode({})
        assert decoded == e

        for v in valid_ids:
            e = self.__entity_class__(id=v)
            decoded = self.__entity_class__.json_decode({'_id': v})
            assert decoded == e
