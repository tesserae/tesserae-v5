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
    is_prose : bool
        Whether text is prose work

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
    is_prose : bool
        Whether text is prose work
    ingestion_status : (str, str)
        Information of whether ingestion has been completed for this text; the
        first item indicates the status; the second item is an accompanying
        message
    ingestion_details : {str: {str: (str, str)}}
        Detailed information about which aspects of ingestion are complete for
        this text. The first level corresponds to a feature type; the second
        level corresponds to a search type; the tuple follows the format of
        ingestion_status

    """

    collection = 'texts'

    def __init__(self,
                 id=None,
                 cts_urn=None,
                 language=None,
                 title=None,
                 author=None,
                 year=None,
                 path=None,
                 is_prose=False,
                 ingestion_status=None,
                 ingestion_details=None,
                 divisions=None):
        super(Text, self).__init__(id=id)
        self.language: typing.Optional[str] = language
        self.title: typing.Optional[str] = title
        self.author: typing.Optional[str] = author
        self.is_prose: typing.Optional[bool] = is_prose
        self.year: typing.Optional[int] = year
        self.path: typing.Optional[str] = path
        self.ingestion_status: tuple[str, str] = tuple(ingestion_status[:2]) \
            if ingestion_status is not None \
            else (TextStatus.INIT, '')
        self.ingestion_details: dict[str, dict[str, tuple[str, str]]] = \
            ingestion_details \
            if ingestion_details is not None \
            else {}
        self.divisions: str = divisions \
            if divisions is not None else []

    def unique_values(self):
        return {
            'language': self.language,
            'title': self.title,
            'author': self.author
        }

    def __repr__(self):
        return (f'Text(language={self.language}, title={self.title}, '
                f'author={self.author}, year={self.year}, '
                f'ingestion_status={self.ingestion_status}, '
                f'ingestion_details={self.ingestion_details}, '
                f'path={self.path}, is_prose={self.is_prose})')

    def update_ingestion_status(self, status_type, msg):
        self.ingestion_status = (status_type, msg)

    def initialize_ingestion_details(self, feature, search_type):
        self.update_ingestion_details(feature, search_type, TextStatus.INIT,
                                      '')

    def update_ingestion_details(self, feature, search_type, status_type, msg):
        if feature not in self.ingestion_details:
            self.ingestion_details[feature]
        self.ingestion_details[feature][search_type] = (status_type, msg)

    def check_ingestion_details(self, feature, search_type):
        """Get the detailed ingestion status of a particular feature for a
        certain search_type

        Returns
        -------
        status_type
            the current status of the ingestion for this feature and search
            type
        msg
            an accompanying message for the ingestion status
        """
        if feature in self.ingestion_details and \
                search_type in self.ingestion_details[feature]:
            return self.ingestion_details[feature][search_type]
        return TextStatus.UNINIT, ''

    def iterate_ingestion_details(self):
        """Iterate through all ingestion information that is not uninitialized

        Yields
        ------
        feature
            the feature that was/is being ingested
        search_type
            whether the feature was/is being ingested for normal Tesserae
            search or for multitext search
        status_type
            the current status of the ingestion for this feature and search
            type
        msg
            an accompanying message for the ingestion status
        """
        for feature, with_feature in self.ingestion_details.items():
            for search_type, (status_type, msg) in with_feature.items():
                yield feature, search_type, status_type, msg


class TextStatus:

    UNINIT = 'Uninitialized'
    INIT = 'Initialized'
    RUN = 'Running'
    DONE = 'Done'
    FAILED = 'Failed'
