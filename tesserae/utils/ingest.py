import time
import traceback

from natsort import natsorted

from tesserae.db.entities import Feature, Token, Unit
from tesserae.db.entities.text import Text, TextStatus
from tesserae.features import get_featurizer
from tesserae.tokenizers import tokenizer_map
from tesserae.unitizer import Unitizer
from tesserae.utils.coordinate import JobQueue
from tesserae.utils.delete import remove_text
from tesserae.utils.multitext import register_bigrams, MULTITEXT_SEARCH
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
    if text.language not in tokenizer_map:
        text.ingestion_status = (TextStatus.FAILED,
                                 'Unknown language: {}'.format(text.language))
        connection.update(text)
        return
    if already_ingested(connection, text):
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
        tokenizer_map[tessfile.metadata.language](connection).tokenize(
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
    _add_feature_for(connection, text, feature, NORMAL_SEARCH)
    if enable_multitext:
        _add_feature_for(connection, text, feature, MULTITEXT_SEARCH)


def _add_feature_for(connection, text, feature, search_type):
    """Add feature information for a text

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to update with a new feature
    feature : str
        The type of feature to extract and account for from the text
    """
    status, _ = text.check_ingestion_details(feature, search_type)
    if status == TextStatus.UNINIT or status == TextStatus.FAILED:
        text.update_ingestion_details(feature, search_type, TextStatus.RUN, '')
        connection.update(text)
        try:
            if search_type == NORMAL_SEARCH:
                _add_feature_for_normal_search(connection, text, feature)
            elif search_type == MULTITEXT_SEARCH:
                _add_feature_for_multitext_search(connection, text, feature)
            else:
                raise ValueError(f'Unknown search type: {search_type}')
            text.update_ingestion_details(feature, search_type,
                                          TextStatus.DONE, '')
            connection.update(text)
        # we want to catch all errors and log them into the Text entity before
        # passing them up
        except Exception as err:
            text.update_ingestion_details(feature, search_type,
                                          TextStatus.FAILED,
                                          traceback.format_exc())
            connection.update(text)
            raise err


def _add_feature_for_normal_search(connection, text, feature):
    """Add feature information for a text for normal search

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to update with a new feature
    feature : str
        The type of feature to extract and account for from the text
    """
    language = text.language
    # use tokens to compute new features and frequencies
    tokens = _get_relevant_tokens(connection, text.id)
    oid_to_form = {
        f.id: f
        for f in connection.find(
            Feature.collection, language=language, feature='form')
    }
    form_oid_to_raw_features = _get_form_oid_to_raw_features(
        oid_to_form, tokens, language, feature)
    _update_features(connection, text, feature, tokens,
                     form_oid_to_raw_features)
    db_feature_cache = {
        f.token: f
        for f in connection.find(
            Feature.collection, language=language, feature=feature)
    }
    # now update tokens and units
    _update_tokens(connection, tokens, feature, db_feature_cache,
                   form_oid_to_raw_features)
    _update_units(connection, text, feature, db_feature_cache,
                  form_oid_to_raw_features, oid_to_form)


def _get_relevant_tokens(connection, text_id):
    """Retrieve only tokens with forms

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text_id: ObjectId
        ObjectId of text

    Returns
    -------
    List[tesserae.db.entities.Token]
    """
    tokens = connection.find(Token.collection, text=text_id)
    return [t for t in tokens if 'form' in t.features and t.features['form']]


def _get_form_oid_to_raw_features(oid_to_form, tokens, language, feature):
    """Map form index to feature tokens extracted from that form

    Parameters
    ----------
    oid_to_form : Dict[ObjectId, tesserae.db.entities.Feature]
        Mapping between ObjectId of a form with its Feature entity
    tokens : list of tesserae.db.entities.Token
        All relevant tokens for a text
    language : str
        The language for which this feature type is being extracted
    feature : str
        The type of feature to extract and account for from the text

    Returns
    -------
    dict[ObjectId, list[str]]
    """
    oid_to_token = {f_id: f.token for f_id, f in oid_to_form.items()}
    form_oids_encountered = list(
        set(token.features['form'] for token in tokens))
    return {
        form_oid: raw_features
        for form_oid, raw_features in zip(
            form_oids_encountered,
            get_featurizer(language, feature)([
                oid_to_token[inner_form_oid]
                for inner_form_oid in form_oids_encountered
            ]))
    }


def _update_features(connection, text, feature, tokens,
                     form_oid_to_raw_features):
    """Update database with Feature information

    As a side effect, feature frequency information is computed.

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to update with a new feature
    feature : str
        The type of feature to extract and account for from the text
    tokens : list of tesserae.db.entities.Token
        All relevant tokens from a text
    form_oid_to_raw_features : dict[ObjectId, list[str]]
        Mapping between form ObjectId and list of strings extracted as raw
        features for that form
    """
    db_feature_cache = {
        f.token: f
        for f in connection.find(
            Feature.collection, language=text.language, feature=feature)
    }
    token_to_features_for_insert, token_to_features_for_update = \
        _calculate_new_and_for_update_features(
            text, feature, db_feature_cache,
            tokens,
            form_oid_to_raw_features)
    connection.insert([f for f in token_to_features_for_insert.values()])
    connection.update([f for f in token_to_features_for_update.values()])
    expected_size = len(token_to_features_for_insert) + \
        len(db_feature_cache)
    wait_limit = 20
    for i in range(wait_limit):
        found = connection.find(Feature.collection,
                                language=text.language,
                                feature=feature)
        if len(found) != expected_size:
            time.sleep(i)
        else:
            return
    raise Exception('Features in database took too long to update; '
                    'aborting feature addition for: '
                    f'{text.author}\t{text.title}')


def _calculate_new_and_for_update_features(text, feature, db_feature_cache,
                                           tokens, form_oid_to_raw_features):
    """Compute new features for insert and old features for update

    In processing the raw features of the given text, feature frequency
    information will be computed.

    Parameters
    ----------
    text : tesserae.db.entities.Text
        Text whose feature frequencies are being analyzed
    feature : str
        Feature type of interest
    db_feature_cache : dict[str, tesserae.db.entities.Feature]
        Mapping between a feature's token and its corresponding Feature entity
    tokens : list of tesserae.db.entities.Token
        All Token entities associated with the text
    form_oid_to_raw_features : dict[ObjectId, list[str]]
        Mapping between form ObjectId and list of strings extracted as raw
        features for that form
    """
    f_token_to_feature_for_insert = {}
    f_token_to_feature_for_update = {}
    text_id_str = str(text.id)
    language = text.language
    for token in tokens:
        form_oid = token.features['form']
        f_tokens = form_oid_to_raw_features[form_oid]
        for f_token in f_tokens:
            if f_token:
                if f_token not in db_feature_cache:
                    if f_token in f_token_to_feature_for_insert:
                        cur_feature = f_token_to_feature_for_insert[f_token]
                        cur_feature.frequencies[text_id_str] += 1
                    else:
                        index = len(db_feature_cache) + \
                            len(f_token_to_feature_for_insert)
                        f_token_to_feature_for_insert[f_token] = Feature(
                            feature=feature,
                            token=f_token,
                            language=language,
                            index=index,
                            frequencies={text_id_str: 1})
                else:
                    if f_token not in f_token_to_feature_for_update:
                        f_token_to_feature_for_update[f_token] = \
                            db_feature_cache[f_token]
                    cur_feature = f_token_to_feature_for_update[f_token]
                    frequencies = cur_feature.frequencies
                    if text_id_str in frequencies:
                        frequencies[text_id_str] += 1
                    else:
                        frequencies[text_id_str] = 1
    return f_token_to_feature_for_insert, f_token_to_feature_for_update


def _update_tokens(connection, tokens, feature, db_feature_cache,
                   form_oid_to_raw_features):
    for token in tokens:
        form_oid = token.features['form']
        token.features[feature] = [
            db_feature_cache[raw_feature].id
            for raw_feature in form_oid_to_raw_features[form_oid]
            if raw_feature in db_feature_cache
        ]
    connection.update(tokens)


def _update_units(connection, text, feature, db_feature_cache,
                  form_oid_to_raw_features, oid_to_form):
    feature_token_to_index = {
        token: f.index
        for token, f in db_feature_cache.items()
    }
    form_index_to_feature_indices = {
        oid_to_form[form_oid].index: [
            feature_token_to_index[raw_feature] for raw_feature in raw_features
            if raw_feature in feature_token_to_index
        ]
        for form_oid, raw_features in form_oid_to_raw_features.items()
    }
    units = connection.find(Unit.collection, text=text.id)
    for unit in units:
        for token in unit.tokens:
            form_index = token['features']['form'][0]
            token['features'][feature] = form_index_to_feature_indices[
                form_index]
    connection.update(units)


def _add_feature_for_multitext_search(connection, text, feature):
    # what this will do remains uncertain
    raise NotImplementedError(
        'Feature addition for multitext not yet implemented')
