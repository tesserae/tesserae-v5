import pytest

import copy

from tesserae.db.entities.match import Match

from entity_test_functions import entity_init, entity_copy, entity_eq, \
                                  entity_id, entity_json_encode, \
                                  entity_json_decode


@pytest.fixture(scope='module')
def default_attrs():
    return {'_id': None, 'units': [], 'tokens': [], 'score': None,
            'metadata': {}}


@pytest.fixture(scope='module')
def valid_attrs(populate):
    return populate['matches']


def test_init(default_attrs, valid_attrs):
    entity_init(Match, default_attrs, valid_attrs)


def test_copy(default_attrs, valid_attrs):
    entity_copy(Match, default_attrs, valid_attrs)


def test_eq(default_attrs, valid_attrs):
    entity_eq(Match, default_attrs, valid_attrs)


def test_id(default_attrs, valid_attrs):
    entity_id(Match, default_attrs, valid_attrs)


def test_json_encode(default_attrs, valid_attrs):
    entity_json_encode(Match, default_attrs, valid_attrs)


def test_json_decode(default_attrs, valid_attrs):
    entity_json_decode(Match, default_attrs, valid_attrs)
