import pytest

import copy

from tesserae.db.entities.frequency import Frequency

from entity_test_functions import entity_init, entity_copy, entity_eq, \
                                  entity_id, entity_json_encode, \
                                  entity_json_decode


@pytest.fixture(scope='module')
def default_attrs():
    return {'_id': None, 'text': None, 'form': None, 'frequency': None}


@pytest.fixture(scope='module')
def valid_attrs(populate):
    return populate['frequencies']


def test_init(default_attrs, valid_attrs):
    entity_init(Frequency, default_attrs, valid_attrs)


def test_copy(default_attrs, valid_attrs):
    entity_copy(Frequency, default_attrs, valid_attrs)


def test_eq(default_attrs, valid_attrs):
    entity_eq(Frequency, default_attrs, valid_attrs)


def test_id(default_attrs, valid_attrs):
    entity_id(Frequency, default_attrs, valid_attrs)


def test_json_encode(default_attrs, valid_attrs):
    entity_json_encode(Frequency, default_attrs, valid_attrs)


def test_json_decode(default_attrs, valid_attrs):
    entity_json_decode(Frequency, default_attrs, valid_attrs)
