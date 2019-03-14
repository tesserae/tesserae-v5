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
    if text.language not in _tokenizers:
        raise ValueError('Unknown language: {}'.format(text.language))
    tessfile = TessFile(text.path, metadata=text)
    tokens, frequencies, feature_sets = \
        _tokenizers[tessfile.metadata.language](connection).tokenize(
            tessfile.read(), text=tessfile.metadata)
    unitizer = Unitizer()
    lines, phrases = unitizer.unitize(tokens, tessfile.metadata)

    result = connection.insert(text)
    text_id = result.inserted_ids[0]
    if feature_sets:
        result = connection.insert(feature_sets)
    if frequencies:
        result = connection.insert(frequencies)
    if lines or phrases:
        result = connection.insert(lines + phrases)
    if tokens:
        result = connection.insert(tokens)
    return text_id
