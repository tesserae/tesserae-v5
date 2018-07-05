import pytest

from tesserae.text_access.storage import retrieve_text_list, insert_text, \
                                         load_text, update_text, NoTextError, \
                                         DuplicateTextError, TextExistsError
from tesserae.db import Text
from tesserae.utils import TessFile

import json
import os
import random


@pytest.fixture(scope='module')
def newfiles(tessfiles):
    newpath = os.path.join(tessfiles, 'new')
    tessfile_list = []
    for root, dirs, files in os.walk(newpath):
        if len(files) > 0 and root.rfind('new'):
            tessfile_list.extend([os.path.join(root, f) for f in files])
    return tessfile_list


@pytest.fixture(scope='module')
def new_populate(tessfiles):
    fpath = os.path.join(tessfiles, 'new', 'test_db_entries_new.json')
    with open(fpath, 'r') as f:
        news = json.load(f)
    for text in news['texts']:
        text['path'] = os.path.join(tessfiles, text['path'])
    return news


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

    # Test retrieving text by existing CTS URN
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
    count = sum([1 if i['year'] == 38 else 0 for i in coll])
    assert len(tl) == count
    for text in tl:
        assert isinstance(text, Text)
        for doc in coll:
            if doc['_id'] == text.id:
                assert text._attributes == doc

    # Test retrieving texts by non-existent year
    tl = retrieve_text_list(connection, year=3007)
    assert len(tl) == 0


def test_insert_text(connection, populate, newfiles, new_populate):
    # TODO: Test with invalid .tess files once a validator is made
    count = connection.texts.count()

    # Test inserting new texts
    for text in new_populate['texts']:
        if '_id' in text:
            tid = text['_id']
            del text['_id']
        else:
            tid = None
        h = text['hash']
        del text['hash']

        result = insert_text(connection, **text)
        assert connection.texts.count() == count + 1
        count += 1

        doc = connection['texts'].find({'_id': result.inserted_id})
        assert doc.count() == 1
        doc = doc[0]
        text['hash'] = h
        for k in text:
            assert doc[k] == text[k]
        assert doc['hash'] == text['hash']

        if tid is not None:
            text['_id'] = tid

    # Clean up to put the DB in its original state
    for text in new_populate['texts']:
        connection.texts.delete_one({'cts_urn': text['cts_urn']})

    # Test inserting existing texts (should fail)
    for text in populate['texts']:
        if '_id' in text:
            tid = text['_id']
            del text['_id']
        else:
            tid = None
        h = text['hash']
        del text['hash']

        with pytest.raises(TextExistsError):
            result = insert_text(connection, **text)

        text['hash'] = h
        if tid is not None:
            text['_id'] = tid

    # Test inserting non-existent texts
    with pytest.raises(FileNotFoundError):
        insert_text(connection, '', '', '', '', 1, [], '/foo/bar.tess')


def test_load_text(connection, populate, new_populate):
    # Test loading texts that exist in the database
    for text in populate['texts']:
        tessfile = load_text(connection, text['cts_urn'])
        assert tessfile.path == text['path']
        assert tessfile.hash == text['hash']

        tessfile = load_text(connection, text['cts_urn'], buffer=False)
        assert tessfile.path == text['path']
        assert tessfile.hash == text['hash']

    # Test loading non-existent texts
    for text in new_populate['texts']:
        print(text)
        with pytest.raises(NoTextError):
            load_text(connection, text['cts_urn'])

    # Test loading non-CTS URNs
    with pytest.raises(NoTextError):
        load_text(connection, 'foo')

    with pytest.raises(NoTextError):
        load_text(connection, 'bar')

    with pytest.raises(NoTextError):
        load_text(connection, 'baz')


def test_update_text(connection, populate, new_populate):
        # Set up some values to use as updates
        languages = ['esperanto', 'gaelic', 'carib', 'afrikaans', 'raptor']
        titles = ['the hitchhiker\'s guide to the galaxy', 's/z',
                  'harry potter and the disengenuous debugger',
                  'godel, escher, bach', 'the lusty argonian maid']
        authors = ['hey you', 'bob', 'mork', 'yeah, you', 'stonework']
        years = [1978, 2285, 10, 354, 2077]

        # Test updating existing texts with random selections from above
        for text in populate['texts']:
            new_vals = {
                'language': languages[random.randint(0, len(languages) - 1)],
                'title': languages[random.randint(0, len(titles) - 1)],
                'author': languages[random.randint(0, len(authors) - 1)],
                'year': languages[random.randint(0, len(years) - 1)],
            }

            result = update_text(connection, cts_urn=text['cts_urn'],
                                 **new_vals)

            # Ensure that the update actually occurred by pulling the entry
            updated = connection.texts.find_one({'cts_urn': text['cts_urn']})

            for k in new_vals:
                assert updated[k] == new_vals[k]
                new_vals[k] = text[k]  # Set up the revert

            # Revert the update to ensure no side-effects occurred
            result = update_text(connection, cts_urn=text['cts_urn'],
                                 **new_vals)

            # Ensure that the revert actually occurred by pulling the entry
            updated = connection.texts.find_one({'cts_urn': text['cts_urn']})

            for k in new_vals:
                assert updated[k] == text[k]

        # Test updating texts that do not exist in the database
        for text in new_populate['texts']:
            with pytest.raises(NoTextError):
                update_text(connection, cts_urn=text['cts_urn'],
                            **new_vals)

            with pytest.raises(NoTextError):
                update_text(connection, cts_urn=text['cts_urn'])
