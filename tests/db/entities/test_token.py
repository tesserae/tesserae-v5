import pytest

import copy

from tesserae.db.entities.token import Token

from entity_test_functions import entity_init, entity_copy, entity_eq, \
                                  entity_id, entity_json_encode, \
                                  entity_json_decode


@pytest.fixture(scope='module')
def default_attrs():
    return {'_id': None, 'text': None, 'index': None, 'display': None,
            'form': None, 'lemmata': [], 'semantic': [], 'sound': []}


@pytest.fixture(scope='module')
def valid_attrs(populate):
    return populate['tokens']


def test_init(default_attrs, valid_attrs):
    entity_init(Token, default_attrs, valid_attrs)


def test_copy(default_attrs, valid_attrs):
    entity_copy(Token, default_attrs, valid_attrs)


def test_eq(default_attrs, valid_attrs):
    entity_eq(Token, default_attrs, valid_attrs)


def test_id(default_attrs, valid_attrs):
    entity_id(Token, default_attrs, valid_attrs)


def test_json_encode(default_attrs, valid_attrs):
    entity_json_encode(Token, default_attrs, valid_attrs)


def test_json_decode(default_attrs, valid_attrs):
    entity_json_decode(Token, default_attrs, valid_attrs)
