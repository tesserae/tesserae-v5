import pytest

import copy

from bson.objectid import ObjectId

from tesserae.db.entities.entity import Entity

from entity_test_functions import entity_init, entity_copy, entity_eq, \
                                  entity_id, entity_json_encode, \
                                  entity_json_decode


@pytest.fixture(scope='module')
def default_attrs():
    return {'_id': None}


@pytest.fixture(scope='module')
def valid_attrs():
    return [{'_id': None},
            {'_id': 'foo'},
            {'_id': ObjectId('0123456789ab0123456789ab')}]


def test_init(default_attrs, valid_attrs):
    entity_init(Entity, default_attrs, valid_attrs)


def test_copy(default_attrs, valid_attrs):
    entity_copy(Entity, default_attrs, valid_attrs)


def test_eq(default_attrs, valid_attrs):
    entity_eq(Entity, default_attrs, valid_attrs)


def test_id(default_attrs, valid_attrs):
    entity_id(Entity, default_attrs, valid_attrs)


def test_json_encode(default_attrs, valid_attrs):
    entity_json_encode(Entity, default_attrs, valid_attrs)


def test_json_decode(default_attrs, valid_attrs):
    entity_json_decode(Entity, default_attrs, valid_attrs)
