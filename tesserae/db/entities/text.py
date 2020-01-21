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
    of the text's creation. The id and path are for internal bookeeping
    purposes.

    Parameters
    ----------
    id : bson.objectid.ObjectId, optional
        Database id of the text. Should not be set locally.
    language : str, optional
        Language the text was written in.
    title : str, optional
        Title of the text.
    author : str, optional
        Full name of the text's author.
    year : int, optional
        Year that the text was written/published.
    path : str
        Path to .tess file

    Attributes
    ----------
    id : str
        Database id of the text. Should not be set locally.
    language : str
        Language the text was written in.
    title : str
        Title of the text.
    author : str
        Full name of the text's author.
    year : int
        Year that the text was written/published.
    path : str
        Path to .tess file

    """

    collection = 'texts'

    def __init__(self, id=None, cts_urn=None, language=None, title=None,
                 author=None, year=None, unit_types=None, path=None,
                 is_prose=True, hash=None, extras=None):
        super(Text, self).__init__(id=id)
        self.language: typing.Optional[str] = language
        self.title: typing.Optional[str] = title
        self.author: typing.Optional[str] = author
        self.year: typing.Optional[int] = year
        self.path: typing.Optional[str] = path

    def unique_values(self):
        return {
            'language': self.language,
            'title': self.title,
            'author': self.author
        }

    def __repr__(self):
        return (
            f'Text(language={self.language}, title={self.title}, '
            f'author={self.author}, year={self.year}, '
            f'path={self.path})'
        )
