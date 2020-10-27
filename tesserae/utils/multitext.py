"""Functionality related to multitext search"""
from collections import defaultdict
import datetime
import glob
import itertools
import os
import sqlite3
import time
import traceback

from bson.objectid import ObjectId
import numpy as np

import tesserae
from tesserae.db.entities import \
    Feature, Match, MultiResult, Search, Text, Unit
from tesserae.db.entities.text import TextStatus
from tesserae.utils.calculations import get_inverse_text_frequencies
from tesserae.utils.retrieve import TagHelper

MULTITEXT_SEARCH = 'multitext'


def submit_multitext(jobqueue, connection, results_id, parallels_uuid,
                     texts_ids_strs, unit_type):
    """Submit a job for multitext search

    Multitext search submitted by this function will always return results
    based on phrases

    Parameters
    ----------
    jobqueue : tesserae.utils.coordinate.JobQueue
    connection : TessMongoConnection
    results_id : str
        UUID to associate with the multitext search to be performed
    parallels_uuid : str
        UUID associated with the Search results on which to run multitext
        search
    texts_ids_str : list of str
        stringified ObjectIds of Texts in which bigrams of the Search results
        are to be searched
    unit_type : str
        unit by which to examine specified texts; if the text is a prose work,
        this option is ignored and the text will be examined by phrase

    """
    parameters = {
        'parallels_uuid': parallels_uuid,
        'text_ids': texts_ids_strs,
        'unit_type': unit_type,
    }
    results_status = Search(results_id=results_id,
                            search_type=MULTITEXT_SEARCH,
                            status=Search.INIT,
                            msg='',
                            parameters=parameters)
    connection.insert(results_status)
    kwargs = {
        'results_status': results_status,
        'parallels_uuid': parallels_uuid,
        'texts_ids_strs': texts_ids_strs,
        'unit_type': unit_type,
    }
    jobqueue.queue_job(_run_multitext, kwargs)


def _run_multitext(connection, results_status, parallels_uuid, texts_ids_strs,
                   unit_type):
    """Runs multitext search

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
    parallels_uuid : str
        UUID associated with the Search results on which to run multitext
        search
    texts_ids_strs : list of str
        stringified ObjectIds of Texts in which bigrams of the Search results
        are to be searched
    unit_type : str
        unit by which to examine specified texts; if the text is a prose work,
        this option is ignored and the text will be examined by phrase

    """
    start_time = time.time()
    try:
        search = connection.find(Search.collection,
                                 results_id=parallels_uuid)[0]
        results_status.update_current_stage_value(0.33)
        connection.update(results_status)
        matches = connection.find(Match.collection, search_id=search.id)
        results_status.update_current_stage_value(0.66)
        connection.update(results_status)
        texts = connection.find(Text.collection,
                                _id=[ObjectId(tid) for tid in texts_ids_strs])
        results_status.update_current_stage_value(1.0)

        results_status.add_new_stage('get multitext data')
        search_id = results_status.id
        results_status.status = Search.RUN
        results_status.last_queried = datetime.datetime.utcnow()
        connection.update(results_status)
        raw_results = multitext_search(results_status, connection, matches,
                                       search.parameters['method']['feature'],
                                       unit_type, texts)
        results_status.update_current_stage_value(1.0)

        results_status.add_new_stage('save multitext results')
        connection.update(results_status)
        stepsize = 5000
        for start in range(0, len(matches), stepsize):
            results_status.update_current_stage_value(start / len(matches))
            connection.update(results_status)
            multiresults = [
                MultiResult(search_id=search_id,
                            match_id=m.id,
                            bigram=list(bigram),
                            units=[v[0] for v in values],
                            scores=[v[1] for v in values])
                for m, result in zip(matches[start:start + stepsize],
                                     raw_results[start:start + stepsize])
                for bigram, values in result.items() if values
            ]
            connection.insert_nocheck(multiresults)

        results_status.update_current_stage_value(1.0)
        results_status.status = Search.DONE
        results_status.msg = 'Done in {} seconds'.format(time.time() -
                                                         start_time)
        results_status.last_queried = datetime.datetime.utcnow()
        connection.update(results_status)
    # we want to catch all errors and log them into the Search entity
    except:  # noqa: E722
        results_status.status = Search.FAILED
        results_status.msg = traceback.format_exc()
        results_status.last_queried = datetime.datetime.utcnow()
        connection.update(results_status)


def compute_tesserae_score(inverse_frequencies, distances):
    """Compute Tesserae score

    Parameters
    ----------
    inverse_frequencies : list of float
    distances : list of int

    Returns
    -------
    float

    """
    return np.log(sum(inverse_frequencies) / sum(distances))


def compute_inverse_frequencies(connection, feature_type, text_id):
    """Compute inverse text frequencies by specified feature type

    Parameters
    ----------
    connection : tesserae.db.mongodb.TessMongoConnection
    feature_type : str
        Feature category to be used in calculating frequencies
    text_id : bson.objectid.ObjectId
        ObjectId of the text whose feature frequencies are to be computed

    Returns
    -------
    1d np.array
        index by form index to obtain corresponding inverse text frequency
    """
    inv_freqs_dict = get_inverse_text_frequencies(connection, feature_type,
                                                  text_id)
    inverse_frequencies = np.zeros(max(inv_freqs_dict) + 1)
    inverse_frequencies[[k for k in inv_freqs_dict.keys()]] = \
        [v for v in inv_freqs_dict.values()]
    return inverse_frequencies


class BigramWriter:
    """Handles writing bigram databases

    Intended to be used in a context (via "with").  Can register bigram data to
    be written to a database.

    Upon closing, a BigramWriter, will flush all remaining data and build
    indices on the databases.
    """

    BIGRAM_DB_DIR = os.path.join(os.path.expanduser('~'), 'tess_data',
                                 'bigrams')
    transaction_threshold = 10000

    def __init__(self, text_id, unit_type):
        """

        As a side-effect, the bigram database directory is created if it does
        not already exist; also, if a bigram database had previously been
        written for this text and unit type, it will be over-written

        Parameters
        ----------
        text_id : ObjectId
            ObjectId of Text
        unit_type : {'line', 'phrase'}
            the type of Unit in which the bigrams are found
        """
        if not os.path.isdir(BigramWriter.BIGRAM_DB_DIR):
            os.makedirs(BigramWriter.BIGRAM_DB_DIR, exist_ok=True)
        for db_path in glob.glob(
                _create_bigram_db_path(text_id, unit_type, '*')):
            os.remove(db_path)
        self.text_id = text_id
        self.unit_type = unit_type
        # dict[feature, list[db_row]]
        self.data = defaultdict(list)

    def record_bigrams(self, feature, positioned_unigrams, forms,
                       inverse_frequencies, unit_id):
        """Compute and store bigram information

        It is assumed that all of the Unit's unigram information for the
        specified feature are passed in

        When enough bigrams have been collected, this BigramWriter will write
        out the data to the appropriate database

        Parameters
        ----------
        feature : str
            the type of feature being recorded
        positioned_unigrams : list of list of int
            the outer list corresponds to a position within the Unit; the inner
            list contains the individual instances of the feature type
        forms : list of list of int
            the outer list corresponds to a position within the Unit; the inner
            list contains the individual instances of the form feature type
        inverse_frequencies : 1d np.array
            mapping from form index number to text frequency for text to which
            the unit associated with ``unit_id`` belongs
        unit_id : ObjectId
            ObjectId of the Unit whose bigrams are being recorded
        """
        collected = {}
        for pos1, (unigrams1,
                   outerform1) in enumerate(zip(positioned_unigrams, forms)):
            form1 = outerform1[0]
            for word1 in unigrams1:
                pos2_start = pos1 + 1
                for pos2_increment, (unigrams2, outerform2) in enumerate(
                        zip(positioned_unigrams[pos2_start:],
                            forms[pos2_start:])):
                    form2 = outerform2[0]
                    for word2 in unigrams2:
                        bigram = tuple(sorted([word1, word2]))
                        if bigram not in collected:
                            collected[bigram] = compute_tesserae_score(
                                inverse_frequencies[[form1, form2]],
                                # distances are computed as expected in
                                # multitext (i.e., two adjacent words have a
                                # distance of 1, words that have one word
                                # between them have a distance of two, etc.)
                                (
                                    pos2_increment + 1, ))
        unit_id_binary = unit_id.binary
        to_write = self.data[feature]
        to_write.extend([(word1, word2, unit_id_binary, score)
                         for (word1, word2), score in collected.items()])

        if len(to_write) > BigramWriter.transaction_threshold:
            self.write_data(feature, to_write)
            to_write.clear()

    def write_data(self, feature, to_write):
        """Write out bigram information to database

        Parameters
        ----------
        feature : str
            the type of feature being recorded
        to_write : list 5-tuple
        """
        dbpath = _create_bigram_db_path(self.text_id, self.unit_type, feature)
        if not os.path.exists(dbpath):
            conn = sqlite3.connect(dbpath)
            conn.execute('create table bigrams('
                         'id integer primary key, '
                         'word1 integer, '
                         'word2 integer, '
                         'unitid blob(12), '
                         'score real '
                         ')')
        else:
            conn = sqlite3.connect(dbpath)
        with conn:
            conn.executemany(
                'insert into bigrams(word1, word2, unitid, score) '
                'values (?, ?, ?, ?)', to_write)
        conn.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for feature, to_write in self.data.items():
            if to_write:
                self.write_data(feature, to_write)
                to_write.clear()
            dbpath = _create_bigram_db_path(self.text_id, self.unit_type,
                                            feature)
            conn = sqlite3.connect(dbpath)
            conn.execute('drop index if exists bigrams_index')
            conn.execute('create index bigrams_index on bigrams ('
                         'word1, word2)')


def _create_bigram_db_path(text_id, unit_type, feature):
    """Create a path to the bigram database for the specified options

    Parameters
    ----------
    text_id : ObjectId
        ObjectId of Text
    unit_type : {'line', 'phrase'}
        the type of Unit in which the bigrams are found
    feature : str
        the type of feature of the bigrams

    Returns
    -------
    str
        the path of the bigram database associated with the specified options

    """
    text_id_str = str(text_id)
    return str(
        os.path.join(BigramWriter.BIGRAM_DB_DIR,
                     f'{text_id_str}_{unit_type}_{feature}.db'))


def register_bigrams(connection, text):
    """Compute and store bigram information for the specified Text

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the Tesserae database
    text_id : tesserae.db.entities.Text
        The text whose bigram information is to be computed and stored

    """
    accepted_features = ('form', 'lemmata')
    for feature in accepted_features:
        text.update_ingestion_details(feature, MULTITEXT_SEARCH,
                                      TextStatus.RUN, '')
    connection.update(text)
    feat2inv_freqs = {
        f_type: compute_inverse_frequencies(connection, f_type, text.id)
        for f_type in accepted_features
    }
    unit_types = ['phrase']
    if not text.is_prose:
        unit_types.append('line')
    for unit_type in unit_types:
        with connection.connection[Unit.collection].find(
            {
                'text': text.id,
                'unit_type': unit_type
            }, {
                '_id': True,
                'tokens': True
            },
                no_cursor_timeout=True) as unit_cursor:
            with BigramWriter(text.id, unit_type) as writer:
                for unit_dict in unit_cursor:
                    by_feature = defaultdict(list)
                    unit_id = unit_dict['_id']
                    tokens = unit_dict['tokens']
                    for t in tokens:
                        for feature, values in t['features'].items():
                            if feature in accepted_features:
                                by_feature[feature].append(values)
                    for feature, values in by_feature.items():
                        writer.record_bigrams(feature, values,
                                              by_feature['form'],
                                              feat2inv_freqs[feature], unit_id)
    for feature in accepted_features:
        text.update_ingestion_details(feature, MULTITEXT_SEARCH,
                                      TextStatus.DONE, '')
    connection.update(text)


def unregister_bigrams(connection, text):
    """Remove bigram data for the specified Text

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the Tesserae database
    text : tesserae.db.entities.Text
        Text whose bigram information is to be removed

    """
    text_id_str = str(text.id)
    for filename in glob.glob(
            os.path.join(BigramWriter.BIGRAM_DB_DIR, f'{text_id_str}_*.db')):
        os.remove(filename)


def lookup_bigrams(text_id, unit_type, feature, bigrams):
    """Looks up bigrams

    Parameters
    ----------
    text_id : ObjectId
        ObjectId associated with the Text of the bigrams to retrieve
    unit_type : {'line', 'phrase'}
        the type of Unit in which to look for bigrams
    feature : str
        the type of feature to which the bigram belongs
    bigrams : iterable of 2-tuple of int
        the bigrams of interest

    Returns
    -------
    dict[tuple[int, int], list[tuple(ObjectId, float)]]
        mapping between bigram and a list of tuples, where each tuple contains
        an ObjectId of a Unit to which the bigram belongs and a score for the
        Unit as calculated by the Tesserae formula

    """
    results = {}
    bigram_db_path = _create_bigram_db_path(text_id, unit_type, feature)
    conn = sqlite3.connect(bigram_db_path)
    with conn:
        for word1, word2 in bigrams:
            int1 = int(word1)
            int2 = int(word2)
            if int1 > int2:
                int1, int2 = int2, int1
            bigram = (int1, int2)
            results[bigram] = [(ObjectId(bytes(row[0])), row[1])
                               for row in conn.execute(
                                   'select unitid, score from bigrams where '
                                   'word1=? and word2=?', bigram)]
    return results


def multitext_search(results_status, connection, matches, feature_type,
                     unit_type, texts):
    """Retrieves Units containing matched bigrams

    Multitext search addresses the question, "Given these matches I've found
    through a Tesserae search, how frequently do these matching words occur
    together in other texts?"

    Multitext search takes the results from a previously completed Search as
    the starting point.  Looking at the Matches found in that Search, it looks
    in the unit of specified Texts for matching bigrams within each Match of
    the Search.

    Parameters
    ----------
    results_status : tesserae.db.entities.Search
    connection : TessMongoConnection
    matches : list of Match
        Match entities from which matched bigrams are taken
    feature_type : {'lemmata', 'form'}
        Feature type of words to search for
    unit_type : {'line', 'phrase'}
        Type of Units to look for; if prose texts are specified in `texts`,
        this option will be ignored and only phrase units will be considered
        for those works
    texts : list of Text
        The Texts whose Units are to be searched

    Returns
    -------
    list of dict[(str, str), list of tuple(ObjectId, float)]
        each dictionary within the list corresponds in index to a match from
        ``matches``; the dictionary contains key-value pairs, where the key is
        a bigram and the value is a list of ObjectIds of Units of type
        ``unit_type`` that contains the bigram specified by the key; Units are
        restricted to those which are found in ``texts``
    """
    language = texts[0].language
    token2index = {
        f.token: f.index
        for f in connection.find(
            Feature.collection, feature=feature_type, language=language)
    }
    results_status.update_current_stage_value(0.25)
    connection.update(results_status)

    bigram_indices = set()
    for m in matches:
        for w1, w2 in itertools.combinations(sorted(m.matched_features), 2):
            bigram_indices.add((token2index[w1], token2index[w2]))
    results_status.update_current_stage_value(0.5)
    connection.update(results_status)

    bigram2units = defaultdict(list)
    for text in texts:
        bigram_data = lookup_bigrams(text.id,
                                     'phrase' if text.is_prose else unit_type,
                                     feature_type, bigram_indices)
        for bigram, data in bigram_data.items():
            bigram2units[bigram].extend([u for u in data])
    results_status.update_current_stage_value(0.75)
    connection.update(results_status)

    return [{
        bigram: bigram2units[(token2index[bigram[0]], token2index[bigram[1]])]
        for bigram in itertools.combinations(sorted(m.matched_features), 2)
    } for m in matches]


def check_cache(connection, parallels_uuid, text_ids_str, unit_type):
    """Check whether multitext results are already in the database

    Parameters
    ----------
    connection : TessMongoConnection
    parallels_uuid : str
        UUID associated with the Search results on which to run multitext
        search
    texts_ids_str : list of str
        stringified ObjectIds of Texts in which bigrams of the Search results
        are to be searched
    unit_type : str
        unit by which to examine specified texts; if the text is a prose work,
        this option is ignored and the text will be examined by phrase

    Returns
    -------
    UUID or None
        If the search results are already in the database, return the
        results_id associated with them; otherwise return None

    Notes
    -----
    Helpful links
        https://docs.mongodb.com/manual/tutorial/query-embedded-documents/
        https://docs.mongodb.com/manual/tutorial/query-arrays/
        https://docs.mongodb.com/manual/reference/operator/query/and/
    """
    found = [
        Search.json_decode(f)
        for f in connection.connection[Search.collection].find({
            'search_type':
            MULTITEXT_SEARCH,
            'parameters.parallels_uuid':
            parallels_uuid,
            '$and': [{
                'parameters.text_ids': {
                    '$all': text_ids_str
                }
            }, {
                'parameters.text_ids': {
                    '$size': len(text_ids_str)
                }
            }],
            'parameters.unit_type':
            unit_type,
        })
    ]
    for s in found:
        if s.status != Search.FAILED:
            return s.results_id
    return None


def get_results(connection, search_id, page_options):
    """Retrieve search results with associated id

    Parameters
    ----------
    connection : TessMongoConnection
    search_id : ObjectId
        ObjectId of Search whose results you are trying to retrieve
    page_options : PageOptions
        specification of which search results to retrieve; these paging options
        are applied to the original Tesserae search results, after which the
        corresponding multitext results are then retrieved

    Returns
    -------
    list of MatchResult
    """
    multisearch = connection.find(Search.collection,
                                  _id=search_id,
                                  search_type=MULTITEXT_SEARCH)[0]
    original_matches = [
        m for m in tesserae.utils.search.retrieve_matches_by_page(
            connection,
            tesserae.utils.search.get_id_by_uuid(
                connection, multisearch.parameters['parallels_uuid']),
            page_options)
    ]
    str_id_to_match = {m['object_id']: m for m in original_matches}
    raw_multiresults = _retrieve_raw_multiresults(connection, search_id,
                                                  original_matches)
    id_to_needed_units = _retrieve_id_to_needed_units(connection,
                                                      raw_multiresults)
    tag_helper = _make_tag_helper_from_units(connection,
                                             id_to_needed_units.values())
    str_match_id_to_multiresults = _get_str_match_id_to_multiresults(
        raw_multiresults)
    return [
        {
            'match':
            str_id_to_match[match['object_id']],
            'cross-ref':
            [] if match['object_id'] not in str_match_id_to_multiresults else [
                {
                    'bigram':
                    mr['bigram'],
                    'units': [
                        {
                            'unit_id': str(u.id),
                            'tag': tag_helper.get_display_tag(u.text, u.tags),
                            'snippet': u.snippet,
                            # TODO implement
                            'highlight': [],
                            'score': score
                        } for u, score in zip((
                            id_to_needed_units[uid]
                            for uid in mr['units']), mr['scores'])
                    ]
                } for mr in str_match_id_to_multiresults[match['object_id']]
            ]
        } for match in original_matches
    ]


def _retrieve_raw_multiresults(connection, search_id, original_matches):
    """Retrieve search results with associated id

    Parameters
    ----------
    connection : TessMongoConnection
    search_id : ObjectId
        ObjectId of Search whose results you are trying to retrieve
    original_matches : list of Dict[str, Any]
        mappings representing Match entities from original Tesserae search on
        which this multitext search was based

    Returns
    -------
    List[Dict[str, Any]]
        Each mapping within the list represents multitext result information as
        follows:
            * 'match_id' : ObjectId
                the Match entity with which a particular multitext result is
                associated
            * 'bigram' : Tuple[str, str]
            * 'units' : List[ObjectId]
                Unit entities found by the multitext search to contain the
                bigram indicated
            * 'scores' : List[float]
                Tesserae scores for each Unit found; items in this list
                correspond with items in the list associated with 'units',
                according to their index position
    """
    results = []
    match_params = {'search_id': search_id}
    original_match_ids = [ObjectId(m['object_id']) for m in original_matches]

    increment = 1000
    start = 0
    while start < len(original_matches):
        end = start + increment
        match_params['match_id'] = {'$in': original_match_ids[start:end]}
        results.extend([
            mr for mr in connection.aggregate(MultiResult.collection,
                                              [{
                                                  '$match': match_params
                                              }, {
                                                  '$project': {
                                                      '_id': False,
                                                      'match_id': True,
                                                      'bigram': True,
                                                      'units': True,
                                                      'scores': True,
                                                  }
                                              }],
                                              encode=False)
        ])
        start = end
    return results


def _retrieve_id_to_needed_units(connection, raw_multiresults):
    """Grab units needed for constructing multitext results

    Parameters
    ----------
    connection : TessMongoConnection
    raw_multiresults : List[Dict[str, Any]]
        list of dictionaries containing basic information about a multitext
        search result; in particular, the key 'units' should have a value of a
        list of ObjectIds corresponding to Units in the database

    Returns
    -------
    Dict[ObjectId, tesserae.db.entities.Unit]
        mapping between an ObjectId and its corresponding Unit entity in the
        database
    """
    result = {}
    needed_ids = list(
        set(uid for mr in raw_multiresults for uid in mr['units']))
    increment = 1000
    start = 0
    while start < len(needed_ids):
        end = start + increment
        result.update((unit.id, unit)
                      for unit in connection.find(Unit.collection,
                                                  _id=needed_ids[start:end]))
        start = end
    return result


def _make_tag_helper_from_units(connection, units):
    """Make TagHelper with texts referenced within ``units``

    Parameters
    ----------
    connection : TessMongoConnection
    units : Iterable[tesserae.db.entities.Unit]

    Returns
    -------
    tesserae.utils.retrieve.TagHelper
    """
    needed_text_ids = set(u.text for u in units)
    needed_texts = connection.find(Text.collection, _id=list(needed_text_ids))
    return TagHelper(connection, needed_texts)


def _get_str_match_id_to_multiresults(raw_multiresults):
    result = {}
    for mr in raw_multiresults:
        match_id = str(mr['match_id'])
        if match_id in result:
            result[match_id].append(mr)
        else:
            result[match_id] = [mr]
    return result
