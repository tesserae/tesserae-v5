from tesserae.tokenizers import GreekTokenizer, LatinTokenizer
from tesserae.unitizer import Unitizer
from tesserae.utils import TessFile


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
    if tessfile.metadata.language not in _tokenizers:
        raise ValueError('Unknown language: {}'.format(tessfile.metadata.language))
    tessfile = TessFile(text.path, metadata=text)
    tokens, frequencies, feature_sets = \
        _tokenizers[tessfile.metadata.language].tokenize(
            tessfile.read(), text=tessfile.metadata)
    unitizer = Unitizer()
    lines, phrases = unitizer.unitize(tokens, tessfile.metadata)

    result = connection.insert(text)
    text_id = result.inserted_ids[0]
    result = connection.insert(feature_sets)
    result = connection.insert(frequencies)
    result = connection.insert(lines + phrases)
    result = connection.insert(tokens)
    return text_id
