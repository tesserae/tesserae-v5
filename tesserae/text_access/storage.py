"""Handle database interactions related to texts.

Functions in this module interact with the database (MongoDB instance and local
file storage) to store and retrieve information about texts in the Tesserae
instance. This should be the first stop for clients when using Tesserae
functionality.

Functions
---------
retrieve_text_list
    Retrieve the list of available texts.
insert_text
    Insert a new text into the database.
load_text
    Open a .tess file for reading.
"""
from tesserae.db import create_filter, convert_to_entity, Text
from tesserae.utils import TessFile


class DuplicateTextError(Exception):
    """Raised when duplicate texts exist in the database"""
    def __init__(self, cts_urn):
        msg = 'Multiple texts {} exist in the database.'.format(cts_urn)
        super(DuplicateTextError, self).__init__(msg)


class NoTextError(Exception):
    """Raised when attempting to access a text that is not in the database"""
    def __init__(self, cts_urn):
        msg = 'Text {} does not exist in the database.'.format(cts_urn)
        msg += ' Are you sure you have the correct CTS URN?'
        super(NoTextError, self).__init__(msg)


class TextExistsError(Exception):
    """Raised when attempting to insert a duplicate text in the database"""
    def __init__(self, cts_urn, hash):
        msg = 'A text {} with hash {} exists in the database.'
        super(TextExistsError, self).__init__(msg.format(cts_urn, hash))


@convert_to_entity(Text)
def retrieve_text_list(client, cts_urn=None, language=None, author=None,
                       title=None, year=None, path=None, hash=None):
    """Retrieve the list of available texts.

    Parameters
    ----------
    client : `pymongo.MongoClient`
        Client connection to the database.
    language : str or list of str, optional
        Filter the list by the supplied language(s).
    author : str or list of str, optional
        Filter the list by the supplied author(s).
    title : str or list of str, optional
        Filter the list by the supplied title(s).
    cts_urn : str or list of str, optional
        Filter the list by the supplied cts_urn(s).
    year : int or pair of int, optional
        Filter the list by the supplied year. If a tuple, limit the list to
        texts from the timespan ``year[0] <= x <= year[1]``.

    Returns
    -------
    texts : list of `tesserae.db.Text`
        The list of texts that match the filter conditions.

    """
    # Create the filter
    filter = create_filter(cts_urn=cts_urn, language=language, author=author,
                           title=title, year=year)

    # Retrieve the texts and put them into a nice format
    docs = client['texts'].find(filter)
    texts = [doc for doc in docs]
    return texts


def insert_text(connection, cts_urn, language, author, title, year, unit_types,
                path):
    """Insert a new text into the database.

    Attempt to insert a new text in the database, sanitized to match the
    fields and data types of existing texts.

    Parameters
    ----------
    cts_urn : str
        Unique collection-level identifier.
    language : str
        Language the text is written in.
    author : str
        Full name of the text author.
    title : str
        Title of the text.
    year : int
        Year of text authorship.
    unit_types : str or list of str
        Valid unit-level delimiters for this text.
    path : str
        Path to the raw text file. May be a remote URL.

    Returns
    -------
    result : `pymongo.InsertOneResult`
        The

    Raises
    ------
    DuplicateTextError
        Raised when attempting to insert a text that already exists in the
        database.

    Notes
    -----
    This function should not be made available to everyone. To properly secure
    the database, ensure that only MongoDB users NOT connected to a public-
    facing client application are able to write to the database. See the
    <MongoDB documentation on role-based access control>_ for more information.

    .. _MongoDB documentation on role-based access control: https://docs.mongodb.com/manual/core/authorization/
    """
    text_file = TessFile(path)
    db_texts = retrieve_text_list(connection, cts_urn=cts_urn,
                                  hash=text_file.hash)

    if len(db_texts) == 0:
        text = Text(cts_urn=cts_urn, language=language, author=author,
                    title=title, year=year, unit_types=unit_types, path=path,
                    hash=text_file.hash)
        result = connection.texts.insert_one(text.json_encode(exclude=['_id']))
        return result
    else:
        raise TextExistsError(cts_urn, hash)


def load_text(client, cts_urn, mode='r', buffer=True):
    """Open a .tess file for reading.

    Parameters
    ----------
    cts_urn : str
        Unique collection-level identifier.
    mode : str
        File open mode ('r', 'w', 'a', etc.)
    buffer : bool
        If True, load file contents into memory on-the-fly. Otherwise, load in
        contents on initialization.

    Returns
    -------
    text : `tesserae.utils.TessFile` or None
        A non-/buffered reader for the file at ``path``. If the file does not
        exit in the database, returns None.

    Raises
    ------
    NoTextError
        Raised when the requested text does not exist in the database.
    """
    # Retrieve text data from the database by CTS URN
    text_objs = retrieve_text_list(client, cts_urn=cts_urn)

    # If more than one text was retrieved, database integrity has been
    # compromised. Raise an exception.
    if len(text_objs) > 1:
        raise DuplicateTextError(cts_urn)

    # Attempt to load the first text in the list of text objects. If the list
    # is empty, raise an excpetion.
    try:
        text = TessFile(text_objs[0].path, mode=mode, buffer=buffer)
    except IndexError:
        raise NoTextError(cts_urn)

    return text


def update_text(client, cts_urn, **kws):
    """Update the metadata for a text.

    Parameters
    ----------
    client : `pymongo.MongoClient`
        Client connection to the database.
    cts_urn : str
        The CTS URN of the text to update.
    language : str, optional
        If supplied, the new languageof the text.
    author : str, optional
        If supplied, the new author of the text.
    title : str, optional
        If supplied, the new title of the text.
    year : int, optional
        If supplied, the new year of authorship/publication.
    path : str, optional
        If supplied, the new path to the text file.
    hash : bool, optional
        If supplied, recompute the hash of the text file.

    Returns
    -------
    result : `pymongo.results.UpdateResult`
        Information about the result of the update operation.

    Raises
    ------
    NoTextError
        Raised when no text with ``cts_urn`` exists in the database.
    DuplicateTextError
        Raise when multiple texts with the same CTS URN are found in the
        database.

    Notes
    -----
    This function will not update the CTS URN of a text. CTS URNs are unique
    identifiers, and changing one indicates a fundamental alteration of the
    text, which should be treated as a new version (and given a unique entry in
    the database).

    If multiple texts with the same CTS URN are discovered in the database, the
    operation will not complete as it is unclear which text to update. This
    indicates an external edit to the database and manual repairs are needed.
    """
    text = retrieve_text_list(cts_urn=cts_urn, **kws)

    if len(text) > 1:
        raise DuplicateTextError(cts_urn, '')
    elif len(text) == 0:
        raise NoTextError(cts_urn)
    else:
        text = text[0]

    update = text.to_json()
    if '_id' in update:
        del update['_id']
    result = client.update_one({'cts_urn': cts_urn}, update)
