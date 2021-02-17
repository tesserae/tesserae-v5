"""Utilities for exporting search results to different file formats.

Modules
-------
csv
json
highlight
xml

Functions
---------
dump
dumps
"""
import importlib
import math

from bson.objectid import ObjectId

from tesserae.db.entities import Search, Text


def get_exporter(file_format):
    """Get the correct exporter for the file type.
    
    Parameters
    ----------
    file_format : {'csv','json','xml'}
        The format to export.

    Returns
    -------
    exporter : module
        The exporter to use, with ``dump`` and ``dumps`` functions defined.
    
    Raises
    ------
    ValueError
        Raised when no exporter exists for ``file_format``.
    """
    # Dynamically import the exporter based on the filename. This keeps the
    # imported modules low and prevents collisions.
    try:
        exporter = importlib.import_module(
            f'tesserae.utils.exports.{file_format.lower()}')
    except ImportError:
        msg = f'''Invalid file format "{file_format}" supplied.
                  Must be one of ["csv", "json", or "xml"]'''
        raise ValueError(msg)

    return exporter


def retrieve_search(connection, search_id):
    """Pull search data from the database.

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        Connection to the MongoDB instance.
    search_id : str or `bson.objectid.ObjectID`
        The database id of the search to serialize.

    Returns
    -------
    search : `tesserae.db.entities.Search`
        Search metadata.
    source : `tesserae.db.entities.Text`
    target : `tesserae.db.entities.Text`
        Source and target text metadata.
    
    Raises
    ------
    ValueError
        Raised when ``search``, ``source``, or ``target`` cannot be found in
        the database.
    RuntimeError
        Raised when ``search`` is not complete.
    """
    if isinstance(search_id, str):
        search_id = ObjectId(search_id)

    # Pull the search and text data from the database.
    try:
        search = connection.find(Search.collection, id=search_id)[0]
    except IndexError:
        raise ValueError(f'No search with id {search_id} found.')

    if search.status.lower() != 'done':
        msg = ''.join([
            f'Search {search_id} is still in-progress.',
            'Try again in a few minutes.'
        ])
        raise RuntimeError(msg)

    try:
        source_id = search.parameters['source']['object_id']
        source = connection.find(Text.collection, id=source_id)[0]
    except IndexError:
        msg = ''.join([
            f'No source text with id {source_id} found.',
            'Was the text deleted?'
        ])
        raise ValueError(msg)

    try:
        target_id = search.parameters['target']['object_id']
        target = connection.find(Text.collection, id=target_id)[0]
    except IndexError:
        msg = ''.join([
            f'No target text with id {target_id} found.',
            'Was the text deleted?'
        ])
        raise ValueError(msg)

    return search, source, target


def dump(connection, search_id, file_format, filename, delimiter=','):
    """Dump a Tesserae search to file.

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        Connection to the MongoDB instance.
    search_id : str or `bson.objectid.ObjectID`
        The database id of the search to serialize.
    file_format : {'csv','json','xml'}
        The format to export.
    filename : str
        Path to the output file.
    delimiter : str, optional
        The row iterm separator. Only used when ``file_format`` is 'csv'.
        Default: ','.
    """
    if isinstance(search_id, str):
        search_id = ObjectId(search_id)

    exporter = get_exporter(file_format)

    search, source, target = retrieve_search(connection, search_id)

    args = (filename, connection, search, source, target)
    kwargs = {}
    if file_format.lower() == 'csv':
        kwargs['delimiter'] = delimiter

    exporter.dump(*args, **kwargs)


def dumps(connection, search_id, file_format, delimiter=','):
    """Dump a Tesserae search to stdout.

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        Connection to the MongoDB instance.
    search_id : str or `bson.objectid.ObjectID`
        The database id of the search to serialize.
    file_format : {'csv','json','xml'}
        The format to export.
    delimiter : str, optional
        The row iterm separator. Only used when ``file_format`` is 'csv'.
        Default: ','.
    """
    if isinstance(search_id, str):
        search_id = ObjectId(search_id)

    exporter = get_exporter(file_format)

    search, source, target = retrieve_search(connection, search_id)

    args = (connection, search, source, target)
    kwargs = {}
    if file_format.lower() == 'csv':
        kwargs['delimiter'] = delimiter

    return exporter.dumps(*args, **kwargs)


def export(connection, search_id, file_format, filepath=None, delimiter=','):
    """Dump a Tesserae search to stdout.

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        Connection to the MongoDB instance.
    search_id : str or `bson.objectid.ObjectID`
        The database id of the search to serialize.
    file_format : {'csv','json','xml'}
        The format to export.
    filepath : str, optional
        Path to the output file. If not provided, the export is returned as
        a string.
    delimiter : str, optional
        The row iterm separator. Only used when ``file_format`` is 'csv'.
        Default: ','.

    Returns
    -------
    results : str
        The string with the results formatted by ``file_format`` and
        ``delimiter`` if applicable. Only returned if ``filepath`` is not
        provided.
    """
    if isinstance(search_id, str):
        search_id = ObjectId(search_id)

    if file_format:
        dump(connection, search_id, file_format, filepath, delimiter=delimiter)
    else:
        return dumps(connection, search_id, file_format, delimiter=delimiter)


__all__ = ['dump', 'dumps', 'export']
