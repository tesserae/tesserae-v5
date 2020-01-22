"""Functions for removing information from the database"""
from tesserae.db.entities import Feature, Match, Search, Token, Unit


def remove_text(connection, text):
    """Removes a text from the database

    More than just removing the Text entity associated with the text, this
    function also removes all other records referencing that Text entity.

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        A connection to the database
    text : tesserae.db.entities.Text
        The text to be ingested

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
        connection.connection[Match.collection].delete_many(
            {'search_id': {'$in': [s.id for s in searches]}}
        )
        connection.delete(searches)

    connection.connection[Feature.collection].update_many(
        {'frequencies.'+str(text_id): {'$exists': True}},
        {'$unset': {'frequencies.'+str(text_id): ""}}
    )

    connection.delete(text)
