import numpy as np

from tesserae.db.entities import Feature


def get_corpus_frequencies(connection, feature, language):
    """Get frequency data for a given feature across a particular corpus

    ``feature`` refers to the kind of feature(s) extracted from the text
    (e.g., lemmata).  "feature type" refers to a group of abstract entities
    that belong to a feature (e.g., "ora" is a feature type of lemmata).
    "feature instance" refers to a particular occurrence of a feature type
    (e.g., the last word of the first line of Aeneid 1 could be derived
    from an instance of "ora"; it is also possible to have been derived
    from an instance of "os" (though Latinists typically reject this
    option)).

    This method finds the frequency of feature types by counting feature
    instances by their type and dividing each count by the total number of
    feature instances found.  Each feature type has an index associated
    with it, which should be used to look up that feature type's frequency
    in the corpus.

    Parameters
    ----------
    connection : tesserae.db.mongodb.TessMongoConnection
    feature : str
        Feature category to be used in calculating frequencies
    language : str
        Language to which the features of interest belong

    Returns
    -------
    np.array
    """
    pipeline = [
        # Get all database documents of the specified feature and language
        # (from the "features" collection, as we later find out).
        {'$match': {'feature': feature, 'language': language}},
        # Transform each document.
        {'$project': {
            # Don't keep their database identifiers,
            '_id': False,
            # but keep their indices.
            'index': True,
            # Also show their frequency,
            'frequency': {
                # which is calculated by summing over all the count values
                # found in the sub-document obtained by checking the
                # "frequencies" key in the original document.
                '$reduce': {
                    'input': {'$objectToArray': '$frequencies'},
                    'initialValue': 0,
                    'in': {'$sum': ['$$value', '$$this.v']}
                }
            }
        }},
        # Finally, sort those transformed documents by their index.
        {'$sort': {'index': 1}}
    ]

    freqs = connection.aggregate(
            Feature.collection, pipeline, encode=False)
    freqs = list(freqs)
    freqs = np.array([freq['frequency'] for freq in freqs])
    return freqs / sum(freqs)



