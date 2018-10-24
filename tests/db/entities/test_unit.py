import pytest

import copy

from tesserae.db.entities.unit import Unit

from entity_test_functions import entity_init, entity_copy, entity_eq, \
                                  entity_id, entity_json_encode, \
                                  entity_json_decode


@pytest.fixture(scope='module')
def default_attrs():
    return {'_id': None, 'text': None, 'index': None, 'unit_type': None,
            'tokens': []}


@pytest.fixture(scope='module')
def valid_attrs(populate):
    return populate['frequencies']


def test_init(default_attrs, valid_attrs):
    entity_init(Unit, default_attrs, valid_attrs)


def test_copy(default_attrs, valid_attrs):
    entity_copy(Unit, default_attrs, valid_attrs)


def test_eq(default_attrs, valid_attrs):
    entity_eq(Unit, default_attrs, valid_attrs)


def test_id(default_attrs, valid_attrs):
    entity_id(Unit, default_attrs, valid_attrs)


def test_json_encode(default_attrs, valid_attrs):
    entity_json_encode(Unit, default_attrs, valid_attrs)


def test_json_decode(default_attrs, valid_attrs):
    entity_json_decode(Unit, default_attrs, valid_attrs)
