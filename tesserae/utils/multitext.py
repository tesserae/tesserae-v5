"""Functionality related to multitext search"""
from collections import defaultdict
import itertools
import os
import sqlite3
import time

from bson.objectid import ObjectId

from tesserae.db.entities import Feature, Unit


class BigramWriter:
    """Handles writing bigram databases

    Intended to be used in a context (via "with").  Can register bigram data to
    be written to a database.

    Upon closing, a BigramWriter, will flush all remaining data and build
    indices on the databases.
    """

    BIGRAM_DB_DIR = os.path.join(
        os.path.expanduser('~'), 'tess_data', 'bigrams')
    transaction_threshold = 10000

    def __init__(self, text_id, unit_type):
        """

        As a side-effect, the bigram database directory is created if it does
        not already exist

        Parameters
        ----------
        text_id : ObjectId
            ObjectId of Text
        unit_type : {'line', 'phrase'}
            the type of Unit in which the bigrams are found
        """
        if not os.path.isdir(BigramWriter.BIGRAM_DB_DIR):
            os.makedirs(BigramWriter.BIGRAM_DB_DIR, exist_ok=True)
        self.text_id = text_id
        self.unit_type = unit_type
        # dict[feature, list[db_row]]
        self.data = defaultdict(list)

    def record_bigrams(self, feature, positioned_unigrams, unit_id):
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
        unit_id : ObjectId
            ObjectId of the Unit whose bigrams are being recorded
        """
        collected = {}
        for pos1, unigrams1 in enumerate(positioned_unigrams):
            for word1 in unigrams1:
                pos2_start = pos1 + 1
                for pos2_increment, unigrams2 in enumerate(
                        positioned_unigrams[pos2_start:]):
                    for word2 in unigrams2:
                        bigram = tuple(sorted([word1, word2]))
                        if bigram not in collected:
                            collected[bigram] = (
                                pos1, pos2_start + pos2_increment)
        unit_id_binary = unit_id.binary
        to_write = self.data[feature]
        to_write.extend([
            (word1, word2, unit_id_binary, pos1, pos2)
            for (word1, word2), (pos1, pos2) in collected.items()
        ])

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
        dbpath = _create_bigram_db_path(
            self.text_id, self.unit_type, feature)
        if not os.path.exists(dbpath):
            conn = sqlite3.connect(dbpath)
            conn.execute('create table bigrams('
                            'id integer primary key, '
                            'word1 integer, '
                            'word2 integer, '
                            'unitid blob(12), '
                            'pos1 integer, '
                            'pos2 integer '
                         ')')
        else:
            conn = sqlite3.connect(dbpath)
        with conn:
            conn.executemany(
                'insert into bigrams(word1, word2, unitid, pos1, pos2) '
                'values (?, ?, ?, ?, ?)',
                to_write
            )
        conn.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        for feature, to_write in self.data.items():
            if to_write:
                self.write_data(feature, to_write)
                to_write.clear()
            dbpath = _create_bigram_db_path(
                self.text_id, self.unit_type, feature)
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
    return str(os.path.join(
        BigramWriter.BIGRAM_DB_DIR, f'{text_id_str}_{unit_type}_{feature}.db'))


def register_bigrams(connection, text_id):
    """Compute and store bigram information for the specified Text

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the Tesserae database
    text_id : ObjectId
        ObjectId of Text

    """
    for unit_type in ['line', 'phrase']:
        with connection.connection[Unit.collection].find(
            {'text': text_id, 'unit_type': unit_type},
            {'_id': True, 'tokens': True},
            no_cursor_timeout=True
        ) as unit_cursor:
            with BigramWriter(text_id, unit_type) as writer:
                for unit_dict in unit_cursor:
                    by_feature = defaultdict(list)
                    unit_id = unit_dict['_id']
                    tokens = unit_dict['tokens']
                    for t in tokens:
                        for feature, values in t['features'].items():
                            by_feature[feature].append(values)
                    for feature, values in by_feature.items():
                        writer.record_bigrams(feature, values, unit_id)


def unregister_bigrams(connection, text_id):
    """Remove bigram data for the specified Text

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the Tesserae database
    text_id : ObjectId
        ObjectId of Text

    """
    text_id_str = str(text_id)
    for filename in glob.glob(os.path.join(
            BigramWriter.BIGRAM_DB_DIR, f'{text_id_str}_*.db')):
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
    dict[tuple[int, int], list[ObjectId]]
        mapping between bigram and ObjectId of List of Units to which
        bigrams belong

    """
    print('Looking up bigram')
    start = time.time()
    results = {}
    bigram_db_path = _create_bigram_db_path(
        text_id, unit_type, feature)
    conn = sqlite3.connect(bigram_db_path)
    with conn:
        for word1, word2 in bigrams:
            int1 = int(word1)
            int2 = int(word2)
            if int1 > int2:
                int1, int2 = int2, int1
            bigram = (int1, int2)
            results[bigram] = [
                ObjectId(bytes(row[0]))
                for row in conn.execute(
                    'select unitid from bigrams where '
                    'word1=? and word2=?',
                    bigram
                )
            ]
    print(len(results))
    print('Ferrying time:', time.time()-start)
    return results


def bigram_search(
        connection, word1_index, word2_index, feature, unit_type, text_id):
    """Retrieves all Units of a specified type containing the specified words

    Parameters
    ----------
    connection : TessMongoConnection
    word1_index, word2_index : int
        Feature index of words to be contained in a Unit
    feature : {'lemmata', 'form'}
        Feature type of words to search for
    unit_type : {'line', 'phrase'}
        Type of Units to look for
    text_id : ObjectId
        The ID of Text in whose Units the bigram is to be searched

    Returns
    -------
    list of Unit
        All Units of the specified texts and ``unit_type`` containing
        both ``word1_index`` and ``word2_index``
    """
    bigram = tuple(sorted((word1_index, word2_index)))
    bigram_data = lookup_bigrams(text_id, unit_type, feature,
                                            [bigram])
    if bigram not in bigram_data:
        return []
    return connection.find(Unit.collection,
                           _id=[u for u in bigram_data[bigram]])


def multitext_search(connection, matches, feature_type, unit_type, texts):
    """Retrieves Units containing matched bigrams

    Parameters
    ----------
    connection : TessMongoConnection
    matches : list of Match
        Match entities from which matched bigrams are taken
    feature_type : {'lemmata', 'form'}
        Feature type of words to search for
    unit_type : {'line', 'phrase'}
        Type of Units to look for
    texts : list of Text
        The Texts whose Units are to be searched

    Returns
    -------
    list of dict[(str, str), list of ObjectId]
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
            Feature.collection, feature=feature_type, language=language)}

    bigram_indices = set()
    for m in matches:
        for w1, w2 in itertools.combinations(sorted(m.matched_features), 2):
            bigram_indices.add((token2index[w1], token2index[w2]))

    bigram2units = defaultdict(list)
    for text in texts:
        bigram_data = lookup_bigrams(
            text.id, unit_type, feature_type, bigram_indices)
        for bigram, data in bigram_data.items():
            print(bigram)
            print(data)
            bigram2units[bigram].extend([
                u for u in data
            ])

    return [
        {
            bigram: bigram2units[
                (token2index[bigram[0]], token2index[bigram[1]])]
            for bigram in itertools.combinations(sorted(m.matched_features), 2)
        }
        for m in matches
    ]
