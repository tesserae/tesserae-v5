import pytest

from tesserae.text_access.storage import retrieve_text_list
from tesserae.db import Text


def test_retrieve_text_list(connection, populate):
    coll = populate['texts']

    # Test retrieving texts with no filter
    # This test pattern is repeated in each block
    tl = retrieve_text_list(connection)
    assert len(tl) == len(coll)  # Ensure the correct number of texts returned
    for text in tl:
        assert isinstance(text, Text)  # Ensure they were converted to objects
        for doc in coll:
            if doc['_id'] == text.id:
                assert text._attributes == doc  # Ensure the attributes match

    # Test retrieving text byh existing CTS URN
    tl = retrieve_text_list(connection,
                            cts_urn='urn:cts:latinLit:phi0917.phi001')
    count = sum([1 if i['cts_urn'] == 'urn:cts:latinLit:phi0917.phi001'
                 else 0 for i in coll])
    assert len(tl) == count
    for text in tl:
        assert isinstance(text, Text)
        for doc in coll:
            if doc['_id'] == text.id:
                assert text._attributes == doc

    # Test retrieving texts by non-existent CTS URN
    tl = retrieve_text_list(connection,
                            cts_urn='urn:tess:testDb:jeff6548.jeff547')
    assert len(tl) == 0

    # Test retrieving texts by existing language
    tl = retrieve_text_list(connection, language='latin')
    count = sum([1 if i['language'] == 'latin' else 0 for i in coll])
    assert len(tl) == count
    for text in tl:
        assert isinstance(text, Text)
        for doc in coll:
            if doc['_id'] == text.id:
                assert text._attributes == doc

    # Test retrieving texts by non-existent language
    tl = retrieve_text_list(connection, language='esperanto')
    assert len(tl) == 0

    # Test retrieving texts by existing author
    tl = retrieve_text_list(connection, author='vergil')
    count = sum([1 if i['author'] == 'vergil' else 0 for i in coll])
    assert len(tl) == count
    for text in tl:
        assert isinstance(text, Text)
        for doc in coll:
            if doc['_id'] == text.id:
                assert text._attributes == doc

    # Test retrieving texts by non-existent author
    tl = retrieve_text_list(connection, author='Donald Knuth')
    assert len(tl) == 0

    # Test retrieving texts by year
    tl = retrieve_text_list(connection, year=38)
    count = sum([1 if i['year'] == '38' else 0 for i in coll])
    assert len(tl) == count
    for text in tl:
        assert isinstance(text, Text)
        for doc in coll:
            if doc['_id'] == text.id:
                assert text._attributes == doc

    # Test retrieving texts by non-existent year
    tl = retrieve_text_list(connection, year=3007)
    assert len(tl) == 0
