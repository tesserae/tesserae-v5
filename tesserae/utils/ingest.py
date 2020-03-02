from tesserae.db.entities import Feature
from tesserae.tokenizers import GreekTokenizer, LatinTokenizer
from tesserae.unitizer import Unitizer
from tesserae.utils.tessfile import TessFile
from tesserae.utils.delete import remove_text


_tokenizers = {
    'greek': GreekTokenizer,
    'latin': LatinTokenizer,
}


def ingest_text(connection, text):
    """Update database with a new text

    ``text`` must not already exist in the database

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to be ingested

    Returns
    -------
    ObjectId
        database identifier for the Text object just added

    Raises
    ------
    ValueError
        Raised when unknown language is encountered
    """
    if text.language not in _tokenizers:
        raise ValueError('Unknown language: {}'.format(text.language))
    tessfile = TessFile(text.path, metadata=text)

    result = connection.insert(text)
    text_id = result.inserted_ids[0]

    tokens, tags, features = \
        _tokenizers[tessfile.metadata.language](connection).tokenize(
            tessfile.read(), text=tessfile.metadata)

    feature_cache = {(f.feature, f.token): f for f in connection.find(
        Feature.collection, language=text.language)}
    features_for_insert = []
    features_for_update = []

    for f in features:
        if (f.feature, f.token) not in feature_cache:
            features_for_insert.append(f)
            feature_cache[(f.feature, f.token)] = f
        else:
            f.id = feature_cache[(f.feature, f.token)].id
            features_for_update.append(f)
    connection.insert(features_for_insert)
    connection.update(features_for_update)

    unitizer = Unitizer()
    lines, phrases, properties = unitizer.unitize(
        tokens, tags, tessfile.metadata)

    connection.insert_nocheck(tokens)
    connection.insert_nocheck(lines + phrases)
    connection.insert_nocheck(properties)
    connection.register_bigrams(text, lines + phrases)

    return text_id


def reingest_text(connection, text):
    """Ingest a text again

    Intended for use in the case of ingestion failure

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to be re-ingested

    Returns
    -------
    ObjectId
        database identifier for the Text object just re-added

    """
    remove_text(connection, text)
    text.id = None
    return ingest_text(connection, text)
