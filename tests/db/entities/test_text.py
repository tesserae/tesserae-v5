import pytest

import copy

from tesserae.db.entities.text import Text

from entity_test_functions import entity_init, entity_copy, entity_eq, \
                                  entity_id, entity_json_encode, \
                                  entity_json_decode


@pytest.fixture(scope='module')
def default_attrs():
    return {'_id': None, 'cts_urn': None, 'language': None, 'title': None,
            'author': None, 'year': None, 'unit_types': [], 'path': None,
            'hash': None}


@pytest.fixture(scope='module')
def valid_attrs(populate):
    return populate['texts']


def test_init(default_attrs, valid_attrs):
    entity_init(Text, default_attrs, valid_attrs)


def test_copy(default_attrs, valid_attrs):
    entity_copy(Text, default_attrs, valid_attrs)


def test_eq(default_attrs, valid_attrs):
    entity_eq(Text, default_attrs, valid_attrs)


def test_id(default_attrs, valid_attrs):
    entity_id(Text, default_attrs, valid_attrs)


def test_json_encode(default_attrs, valid_attrs):
    entity_json_encode(Text, default_attrs, valid_attrs)


def test_json_decode(default_attrs, valid_attrs):
    entity_json_decode(Text, default_attrs, valid_attrs)
