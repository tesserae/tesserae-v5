import time
import traceback

from natsort import natsorted

from tesserae.db.entities import Feature, Unit
from tesserae.db.entities.text import Text, TextStatus
from tesserae.features import get_featurizer
from tesserae.tokenizers import GreekTokenizer, LatinTokenizer
from tesserae.tokenizers.base import create_features
from tesserae.unitizer import Unitizer
from tesserae.utils.coordinate import JobQueue
from tesserae.utils.delete import remove_text
from tesserae.utils.multitext import register_bigrams
from tesserae.utils.search import NORMAL_SEARCH
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
    status, msg = text.ingestion_status
    if status == TextStatus.FAILED:
        remove_text(connection, text)
        raise ValueError(msg)
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

    features_ingested = {feature for feature in lines[0].tokens[0]['features']}
    for feature in features_ingested:
        text.update_ingestion_details(feature, NORMAL_SEARCH, TextStatus.DONE,
                                      '')
    connection.update(text)

    connection.insert_nocheck(tokens)
    connection.insert_nocheck(lines + phrases)
    if enable_multitext:
        register_bigrams(connection, text)


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


def add_feature(connection, text, feature, enable_multitext=False):
    """Add feature information for a text

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to update with a new feature
    feature : str
        The type of feature to extract and account for from the text
    enable_multitext : bool (default: False)
        Whether to enable multitext search with this text
    """
    # TODO add check to make sure feature hasn't already been computed for this
    # text
    index_to_form = {
        f.index: f.token
        for f in connection.find(
            Feature.collection, language=text.language, feature='form')
    }
    # use lines to compute new features
    _update_lines_and_features(connection, text, feature, index_to_form)
    db_feature_cache = {
        f.token: f
        for f in connection.find(
            Feature.collection, language=text.language, feature=feature)
    }
    # update other entities, but don't count frequencies again
    update_phrases(connection, text, feature, index_to_form, db_feature_cache)
    update_tokens(connection, text, feature, index_to_form, db_feature_cache)

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
    connection.update(unit)
    # TODO indicate that feature has been computed


def _update_lines_and_features(connection, text, feature, index_to_form):
    """Compute features and update both lines and features in the database

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to update with a new feature
    feature : str
        The type of feature to extract and account for from the text
    index_to_form : dict[int, str]
        Mapping between form index and form token
    """
    lines = connection.find(Unit.collection, text=text.id, unit_type='line')
    form_indices_encountered = list(
        set(token['features']['form'][0] for line in lines
            for token in line.tokens))
    form_index_to_raw_features = {
        form_index: raw_features
        for form_index, raw_features in zip(
            form_indices_encountered,
            get_featurizer(text.language, feature)([
                index_to_form[form_index]
                for form_index in form_indices_encountered
            ]))
    }
    _update_features(form_index_to_raw_features)
    db_feature_cache = {
        f.token: f
        for f in connection.find(
            Feature.collection, language=text.language, feature=feature)
    }


def _update_features(connection, text, feature, form_index_to_raw_features):
    """Update database with Feature information

    This is where frequency information is computed.

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to update with a new feature
    feature : str
        The type of feature to extract and account for from the text
    form_index_to_raw_features : dict[int, list[str]]
        Mapping between form index and list strings extracted as raw features
        for that form
    """
    language = text.language
    text_id_str = str(text.id)
    db_feature_cache = {
        f.token: f
        for f in connection.find(
            Feature.collection, language=language, feature=feature)
    }
    token_to_features_for_insert = {}
    token_to_features_for_update = {}
    for values in form_index_to_raw_features.values():
        for token in values:
            if token:
                if token not in db_feature_cache:
                    if token not in token_to_features_for_insert:
                        index = len(db_feature_cache) + \
                            len(token_to_features_for_insert)
                        token_to_features_for_insert[token] = Feature(
                            feature=feature,
                            token=token,
                            language=language,
                            index=index,
                            frequencies={text_id_str: 1})
                    else:
                        cur_feature = token_to_features_for_insert[token]
                        cur_feature.frequencies[text_id_str] += 1
                else:
                    if token not in token_to_features_for_update:
                        token_to_features_for_update[token] = \
                            db_feature_cache[token]
                    cur_feature = token_to_features_for_update[token]
                    frequencies = cur_feature.frequencies
                    if text_id_str not in frequencies:
                        frequencies[text_id_str] = 1
                    else:
                        frequencies[text_id_str] += 1
    connection.insert([f for f in token_to_features_for_insert.values()])
    connection.update([f for f in token_to_features_for_update.values()])


def get_updated_units(connection, text, feature, unit_type):
    """Compute features and updated units for the text

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to update with a new feature
    feature : str
        The type of feature to extract and account for from the text
    unit_type : str
        The type of unit

    Returns
    -------
    list of Unit
        Every Unit in this list is associated with ``text`` and contains the
        new feature information
    """
    units = connection.find(Unit.collection, text=text.id, unit_type=unit_type)


def gen_updated_unit_and_features(connection, text, feature):
    """Compute features and updated units for the text

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to update with a new feature
    feature : str
        The type of feature to extract and account for from the text
    """
    units = connection.find(Unit.collection, text=text, unit_type=unit_type)
    # go by line; keep track of frequencies of new features
    # then go by phrase; don't double-count frequencies
    for unit in units:
        features_already_in_db = connection.find(Feature.collection,
                                                 language=text.language,
                                                 feature=feature)
        feature_list = extract_new_features(units, text.language, feature)
        # TODO make sure that feature_list really is what I think it is
        for token, extracted in zip(unit.tokens, feature_list):
            token['features'][feature] = [f.index for f in extracted]
        new_features, _ = create_features(features_already_in_db, text.id,
                                          text.language, feature, feature_list)
        yield unit, new_features


def extract_new_features(units, language, feature):
    """Extract new features from specified units

    Parameters
    ----------
    units : list of tesserae.db.entities.Unit
        Units from which features are to be extracted
    language : str
        The language of the text from which the units come
    feature : str
        The type of feature to extract and account for from the text
    """
    # TODO figure out how to extract appropriate features
