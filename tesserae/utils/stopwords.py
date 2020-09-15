"""Functions for interfacing stopwords with database information"""
import numpy as np

from tesserae.db.entities import Entity, Feature


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
    pipeline = [{
        '$match': {
            'language': language,
            'feature': feature_type,
            'token': {
                '$in': stopwords
            }
        }
    }, {
        '$project': {
            '_id': False,
            'index': True
        }
    }]

    stoplist = conn.aggregate(Feature.collection, pipeline, encode=False)
    return np.array([s['index'] for s in stoplist], dtype=np.uint32)


def create_stoplist(connection, n, feature, language, basis='corpus'):
    """Compute a stoplist of `n` tokens.

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
    n : int
        The number of tokens to include in the stoplist.
    feature : str
        The type of feature to consider
    language : str
        The language of the stopwords to look for
    basis : list of tesserae.db.entities.Text or 'corpus'
        The texts to use as the frequency basis. If 'corpus', use frequencies
        across the entire corpus.

    Returns
    -------
    stoplist : 1d np.array of np.unit32
        The `n` most frequent tokens in the basis texts.
    """
    pipeline = [
        {
            '$match': {
                'feature': feature,
                'language': language
            }
        },
    ]

    if basis == 'corpus':
        pipeline.append({
            '$project': {
                '_id': False,
                'index': True,
                'token': True,
                'frequency': {
                    '$reduce': {
                        'input': {
                            '$objectToArray': '$frequencies'
                        },
                        'initialValue': 0,
                        'in': {
                            '$sum': ['$$value', '$$this.v']
                        }
                    }
                }
            }
        })
    else:
        basis = [t.id if isinstance(t, Entity) else t for t in basis]
        pipeline.extend([{
            '$project': {
                '_id': False,
                'index': True,
                'token': True,
                'frequency': {
                    '$sum': ['$frequencies.' + str(t_id) for t_id in basis]
                }
            }
        }])

    pipeline.extend([{
        '$sort': {
            'frequency': -1
        }
    }, {
        '$limit': n
    }, {
        '$project': {
            'token': True,
            'index': True,
            'frequency': True
        }
    }])

    stoplist = connection.aggregate(Feature.collection, pipeline, encode=False)
    stoplist = list(stoplist)
    return np.array([s['index'] for s in stoplist], dtype=np.uint32)


def get_stoplist_indices(connection, stopwords, feature=None, language=None):
    """Retrieve feature indices for the given words

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
    stopwords : list of str
        Words to consider as stopwords; these must be in normalized form
    feature : str
        The type of feature to consider
    language : str
        The language of the stopwords to look for

    Returns
    -------
    stoplist : 1d np.array of np.unit32
        The `n` most frequent tokens in the basis texts.
    """
    pipeline = [{
        '$match': {
            'token': {
                '$in': stopwords
            }
        }
    }, {
        '$project': {
            '_id': False,
            'index': True
        }
    }]

    if language is not None:
        pipeline[0]['$match']['language'] = language

    if feature is not None:
        pipeline[0]['$match']['feature'] = feature

    stoplist = connection.aggregate(Feature.collection, pipeline, encode=False)
    return np.array([s['index'] for s in stoplist], dtype=np.uint32)


def get_stoplist_tokens(connection, stopword_indices, feature, language):
    """Retrieve words for the given indices

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
    stopword_indices : 1d np.array of np.uint32
        Feature indices of stopwords
    feature : str
        The feature type of the stopwords
    language : str
        The language of the stopwords

    Returns
    -------
    stoplist : list of str
        The `n` most frequent tokens in the basis texts.
    """
    results = connection.find(Feature.collection,
                              index=[int(i) for i in stopword_indices],
                              language=language,
                              feature=feature)
    results = {f.index: f.token for f in results}
    return [results[i] for i in stopword_indices]
