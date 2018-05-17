import pytest

from tesserae.db.entities import convert_to_entity, Entity, Text, Unit, \
                                 Token, NGram, Match

from bson.objectid import ObjectId


def create_instances(count=1, **kws):
    return kws if count == 1 else [kws for _ in range(count)]


def test_convert_to_entity():
    classes = [TestText, TestUnit, TestToken, TestNGram, TestMatch]

    for c in classes:
        # Create a non-parameterized (null fields) and parameterized instance
        n = c.__entity_class__()
        p = c.__entity_class__(**c.__test_kwargs__)

        # Test converting one non-parameterized instance
        r = convert_to_entity(c.__entity_class__)(create_instances)()
        assert isinstance(r, c.__entity_class__)
        assert r == n

        # Test converting one parameterized instance
        r = convert_to_entity(c.__entity_class__)(create_instances)(
            **c.__test_kwargs__)
        assert isinstance(r, c.__entity_class__)
        assert r == p

        # Test converting one non-parameterized instance
        r = convert_to_entity(c.__entity_class__)(create_instances)(count=100)
        assert len(r) == 100
        for o in r:
            assert isinstance(o, c.__entity_class__)
            assert o == n

        # Test converting one parameterized instance
        r = convert_to_entity(c.__entity_class__)(create_instances)(
            count=100, **c.__test_kwargs__)
        assert len(r) == 100
        for o in r:
            assert isinstance(o, c.__entity_class__)
            assert o == p

    # Ensure that invalid classes are not considered
    invalids = [Entity, int, float, str, bool, None, list, dict, set, tuple]
    for i in invalids:
        with pytest.raises(TypeError):
            print(i)
            convert_to_entity(i)(create_instances)()


class TestEntity(object):
    __entity_class__ = Entity

    def test_init(self):
        # Test the default instantiation
        e = self.__class__.__entity_class__()
        assert hasattr(e, '_attributes')
        assert e._attributes == {}


class TestText(TestEntity):
    __entity_class__ = Text
    __test_kwargs__ = {'id': str(ObjectId(b'thisisatest!')),
                       'language': 'latin',
                       'cts_urn': 'urn:cts:latinLit:phi0690.phi002',
                       'author': 'Vergil',
                       'title': 'Aeneid',
                       'year': 19,
                       'unit_types': ['line', 'phrase']}

    def test_init(self):
        # Test the default instantiation
        e = self.__class__.__entity_class__()
        assert e.id is None
        assert e.language is None
        assert e.cts_urn is None
        assert e.title is None
        assert e.author is None
        assert e.year is None
        assert e.unit_types == []

        # Test instantiation with test args
        e = self.__class__.__entity_class__(**self.__class__.__test_kwargs__)
        assert e.id == str(ObjectId(b'thisisatest!'))
        assert e.language == 'latin'
        assert e.cts_urn == 'urn:cts:latinLit:phi0690.phi002'
        assert e.title == 'Aeneid'
        assert e.author == 'Vergil'
        assert e.year == 19
        assert e.unit_types == ['line', 'phrase']

    def test_id(self):
        pass

    def test_cts_urn(self):
        pass

    def test_language(self):
        pass

    def test_title(self):
        pass

    def test_author(self):
        pass

    def test_year(self):
        pass

    def test_unit_types(self):
        pass


class TestUnit(TestEntity):
    __entity_class__ = Unit
    __test_kwargs__ = {'id': str(ObjectId(b'thisisatest!'))}

    def test_init(self):
        # Test the default instantiation
        e = self.__class__.__entity_class__()

    def test_id(self):
        pass


class TestToken(TestEntity):
    __entity_class__ = Token
    __test_kwargs__ = {'id': str(ObjectId(b'thisisatest!'))}

    def test_init(self):
        # Test the default instantiation
        e = self.__class__.__entity_class__()

    def test_id(self):
        pass


class TestNGram(TestEntity):
    __entity_class__ = NGram
    __test_kwargs__ = {'id': str(ObjectId(b'thisisatest!'))}

    def test_init(self):
        # Test the default instantiation
        e = self.__class__.__entity_class__()

    def test_id(self):
        pass


class TestMatch(TestEntity):
    __entity_class__ = Match
    __test_kwargs__ = {'id': str(ObjectId(b'thisisatest!'))}

    def test_init(self):
        # Test the default instantiation
        e = self.__class__.__entity_class__()

    def test_id(self):
        pass
