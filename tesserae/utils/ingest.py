import time
import traceback

from natsort import natsorted

from tesserae.db.entities import Feature
from tesserae.db.entities.text import Text, TextStatus
from tesserae.tokenizers import GreekTokenizer, LatinTokenizer
from tesserae.unitizer import Unitizer
from tesserae.utils.coordinate import JobQueue
from tesserae.utils.delete import remove_text
from tesserae.utils.multitext import register_bigrams
from tesserae.utils.search import NORMAL_SEARCH
from tesserae.utils.multitext import MULTITEXT_SEARCH
from tesserae.utils.tessfile import TessFile


class IngestQueue(JobQueue):
    def __init__(self, db_cred):
        # make sure that only one text is ingested at a time
        super().__init__(1, db_cred)


def submit_ingest(ingest_queue, connection, text, file_location):
    """Submit a job for ingesting a text

    Parameters
    ----------
    ingest_queue : IngestQueue
    connection : TessMongoConnection
    text : tesserae.db.entities.Text
        Text entity to be ingested
    file_location : str
        Path to .tess file to be ingested

    Returns
    -------
    ObjectId
        ID of text to be ingested

    """
    connection.insert(text)
    kwargs = {'text': text, 'file_location': file_location}
    ingest_queue.queue_job(_run_ingest, kwargs)
    return text.id


def _run_ingest(connection, text, file_location, enable_multitext=False):
    """Instructions for running ingestion

    Parameters
    ----------
    connection : TessMongoConnection
    text : tesserae.db.entities.Text
        Text entity to be ingested
    file_location : str
        Path to .tess file to be ingested
    enable_multitext : bool (default: False)
        Whether to enable multitext search with this text

    """
    start_time = time.time()
    if text.language not in _tokenizers:
        text.ingestion_status = (TextStatus.FAILED,
                                 'Unknown language: {}'.format(text.language))
        connection.update(text)
        return
    if already_ingested(connection, text):
        text.ingestion_status = (
            TextStatus.FAILED,
            f'Text already in database (author: {text.author}, '
            f'title: {text.title})')
        connection.update(text)
        return

    text.path = file_location
    tessfile = TessFile(text.path, metadata=text)

    try:
        _ingest_tessfile(connection, text, tessfile, enable_multitext)
        text.ingestion_status = (
            TextStatus.DONE,
            f'Ingestion complete in {time.time()-start_time} seconds')
        connection.update(text)
    # we want to catch all errors and log them into the Text entity
    except:  # noqa: E722
        text.ingestion_status = (TextStatus.FAILED, traceback.format_exc())
        connection.update(text)


_tokenizers = {
    'greek': GreekTokenizer,
    'latin': LatinTokenizer,
}


def already_ingested(connection, text):
    """Query database to see if a text is already ingested

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text whose ingestion status is in question

    Returns
    -------
    bool
    """
    found = connection.find(Text.collection,
                            author=text.author,
                            title=text.title)
    if found and found[0].ingestion_status == TextStatus.DONE:
        return True
    return False


def ingest_text(connection, text, enable_multitext=False):
    """Update database with a new text

    ``text`` must not already exist in the database

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to be ingested
    enable_multitext : bool (default: False)
        Whether to enable multitext search with this text

    Returns
    -------
    ObjectId
        database identifier for the Text object just added

    Raises
    ------
    ValueError
        Raised when an error occurs during ingestion
    """
    connection.insert(text)
    _run_ingest(connection, text, text.path, enable_multitext)
    if text.ingestion_status == TextStatus.FAILED:
        error_msg = text.ingestion_msg
        remove_text(connection, text)
        raise ValueError(error_msg)
    return text.id


def _ingest_tessfile(connection, text, tessfile, enable_multitext=False):
    """Process .tess file for inclusion in Tesserae database

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        Text entity associated with the .tess file to be ingested; must
        already be added to Text.collection but not yet ingested
    tessfile : tesserae.utils.TessFile
        .tess file to be ingested
    enable_multitext : bool (default: False)
        Whether to enable multitext search with this text
    """
    tokens, tags, features = \
        _tokenizers[tessfile.metadata.language](connection).tokenize(
            tessfile.read(), text=tessfile.metadata)

    text.divisions = _extract_divisions(tags)
    connection.update(text)

    feature_cache = {
        (f.feature, f.token): f
        for f in connection.find(Feature.collection, language=text.language)
    }
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
    lines, phrases = unitizer.unitize(tokens, tags, tessfile.metadata)

    connection.insert_nocheck(tokens)
    connection.insert_nocheck(lines + phrases)
    if enable_multitext:
        register_bigrams(connection, text.id)


def _extract_divisions(tags):
    if not tags or len([v for v in tags[0].split('.') if v]) < 2:
        return []
    return natsorted(list({tag.split('.')[0] for tag in tags}))


def reingest_text(connection, text, enable_multitext=False):
    """Ingest a text again

    Intended for use in the case of ingestion failure

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to be re-ingested
    enable_multitext : bool (default: False)
        Whether to enable multitext search with this text

    Returns
    -------
    ObjectId
        database identifier for the Text object just re-added

    """
    remove_text(connection, text)
    text.id = None
    text.ingestion_complete = False
    return ingest_text(connection, text, enable_multitext)
