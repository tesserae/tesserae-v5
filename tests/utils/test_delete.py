import pytest

from tesserae.db import TessMongoConnection
from tesserae.db.entities import Feature, Text, Token, Unit
from tesserae.utils import ingest_text, remove_text


@pytest.fixture
def removedb(mini_latin_metadata):
    conn = TessMongoConnection('localhost', 27017, None, None, 'removedb')
    for metadata in mini_latin_metadata:
        text = Text.json_decode(metadata)
        ingest_text(conn, text)
    yield conn
    for coll_name in conn.connection.list_collection_names():
        conn.connection.drop_collection(coll_name)


def test_remove(removedb, mini_latin_metadata):
    texts = removedb.find(
        Text.collection,
        title=[m['title'] for m in mini_latin_metadata]
    )

    text_id = texts[0].id
    remove_text(removedb, texts[0])

    tokens = removedb.find(Token.collection)
    assert all([t.text != text_id for t in tokens])

    units = removedb.find(Unit.collection)
    assert all([u.text != text_id for u in units])

    features = removedb.find(Feature.collection)
    assert all([str(text_id) not in f.frequencies for f in features])
