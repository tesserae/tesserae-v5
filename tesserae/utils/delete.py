"""Functions for removing information from the database"""
import os
import shutil

from tesserae.db.entities import \
    Feature, Match, MultiResult, Search, Token, Unit
from tesserae.utils.multitext import \
    BigramWriter, MULTITEXT_SEARCH, unregister_bigrams
from tesserae.utils.search import NORMAL_SEARCH


def remove_results(connection, searches):
    """Remove results in the database associated with a collection of searches

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    searches : list of tesserae.db.entities.Search
        A collection of searches to be deleted, with their associated data

    """
    normal_searches = []
    multi_searches = []
    for search in searches:
        if search.search_type == NORMAL_SEARCH:
            normal_searches.append(search)
        elif search.search_type == MULTITEXT_SEARCH:
            multi_searches.append(search)
    if normal_searches:
        matchdb = connection.connection[Match.collection]
        matchdb.delete_many(
            {'search_id': {'$in': [s.id for s in searches]}}
        )
        # make sure that multitext searches that are built on top of the
        # searches that are about to be deleted are also included in the
        # multitext searches that are to be deleted
        multi_searches.extend(connection.aggregate(
            Search.collection,
            [
                {
                    '$match': {
                        'parameters.search_uuid': {
                            '$in': [s.results_id for s in searches]
                        }
                    }
                }
            ]
        ))
        connection.delete(normal_searches)
    if multi_searches:
        multidb = connection.connection[MultiResult.collection]
        multidb.delete_many(
            {'search_id': {'$in': [m.id for m in multi_searches]}}
        )
        connection.delete(multi_searches)


def remove_text(connection, text):
    """Removes a text from the database

    More than just removing the Text entity associated with the text, this
    function also removes all other records referencing that Text entity.

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to be removed

    """
    text_id = text.id

    connection.connection[Token.collection].delete_many({'text': text_id})
    connection.connection[Unit.collection].delete_many({'text': text_id})

    searches = connection.aggregate(
        Search.collection,
        [
            {
                '$match': {
                    '$or': [
                        {'parameters.source.object_id': str(text_id)},
                        {'parameters.target.object_id': str(text_id)},
                        {'parameters.text_ids': str(text_id)},
                    ]
                }
            }
        ]
    )
    remove_results(connection, searches)

    connection.connection[Feature.collection].update_many(
        {'frequencies.'+str(text_id): {'$exists': True}},
        {'$unset': {'frequencies.'+str(text_id): ""}}
    )

    unregister_bigrams(connection, text_id)

    connection.delete(text)


def obliterate(connection):
    """VERY DANGEROUS! Completely removes the database

    Also removes other files associated with the database (like bigram
    databases)

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database

    """
    if os.path.isdir(BigramWriter.BIGRAM_DB_DIR):
        shutil.rmtree(BigramWriter.BIGRAM_DB_DIR)
    for coll_name in connection.connection.list_collection_names():
        connection.connection.drop_collection(coll_name)
