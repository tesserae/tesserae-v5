"""Database standardization for text metadata.

Classes
-------
Text
    Text metadata data model.
"""
import typing

from tesserae.db.entities import Entity


class Text(Entity):
    """Metadata about a text available to Tesserae.

    Text entries in the Tesserae database contain metadata about text files
    available to Tesserae. The language, title, author, and year are attributes
    of the text's creation. The id, CTS URN, hash, path, and unit types are all
    for internal bookeeping purposes.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    cts_urn : str, optional
        Uniform resource name following the Canonical Text Services convention.
        Unique identifier for the text.
    language : str, optional
        Language the text was written in.
    title : str, optional
        Title of the text.
    author : str, optional
        Full name of the text's author.
    year : int, optional
        Year that the text was written/published.
    unit_types : str or list of str, optional
        Available methods for splitting a text into units.

    Attributes
    ----------
    id : str
        Database id of the text. Should not be set locally.
    cts_urn : str
        Uniform resource name following the Canonical Text Services convention.
        Unique identifier for the text.
    language : str
        Language the text was written in.
    title : str
        Title of the text.
    author : str
        Full name of the text's author.
    year : int
        Year that the text was written/published.
    unit_types : list of str
        Available methods for splitting a text into units.

    """

    collection = 'texts'

    def __init__(self, id=None, cts_urn=None, language=None, title=None,
                 author=None, year=None, unit_types=None, path=None,
                 hash=None):
        super(Text, self).__init__(id=id)
        self.cts_urn: typing.Optional[str] = cts_urn
        self.language: typing.Optional[str] = language
        self.title: typing.Optional[str] = title
        self.author: typing.Optional[str] = author
        self.year: typing.Optional[int] = year
        self.unit_types: typing.List[str] = \
            unit_types if unit_types is not None else []
        self.path: typing.Optional[str] = path
        self.hash: typing.Optional[str] = hash
