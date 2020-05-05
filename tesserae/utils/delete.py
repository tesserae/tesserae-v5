"""Functions for removing information from the database"""
import os
import shutil

from tesserae.db.entities import Feature, Match, Search, Token, Unit
from tesserae.utils.multitext import BigramWriter, unregister_bigrams


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
                '$match': {'texts': text_id}
            }
        ]
    )
    if searches:
        matchdb = connection.connection[Match.collection]
        matchdb.delete_many(
            {'search_id': {'$in': [s.id for s in searches]}}
        )
        # remember to re-index after removing Match entities
        matchdb.reindex()
        connection.delete(searches)

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
