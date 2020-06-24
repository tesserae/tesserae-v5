"""Functions for interfacing stopwords with database information"""
import numpy as np

from tesserae.db.entities import Feature


def get_feature_indices(conn, language, feature_type, stopwords):
    """Retrieve Feature indicies for specified stopwords

    Parameters
    ----------
    conn : TessMongoConnection
    feature_type : str
        The category of Feature the stopwords are supposed to be
    language : str
        The language of the stopwords
    stopwords : list of str
        Normalized tokens to be considered stopwords

    Returns
    -------
    1d np.array of int
        The feature indices of the specified stopwords
    """
    pipeline = [
        {'$match': {
            'language': language,
            'feature': feature_type,
            'token': {'$in': stopwords}
        }},
        {'$project': {
            '_id': False,
            'index': True
        }}
    ]

    stoplist = conn.aggregate(
        Feature.collection, pipeline, encode=False)
    return np.array([s['index'] for s in stoplist], dtype=np.uint32)
