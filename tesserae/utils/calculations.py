from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

from tesserae.db.entities import Feature, Unit


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


def get_feature_counts_by_text(connection, feature, text):
    """Get number of times instances of given feature type occur in a
    particular text


    This method assumes that for a given word type, the feature types
    extracted from any one instance of the word type will be the same as
    all other instances of the same word type.  Thus, further work would be
    necessary, for example, if features could be extracted based on word
    position.

    Parameters
    ----------
    connection : tesserae.db.mongodb.TessMongoConnection
    feature : str
        Feature category to be used in calculating frequencies
    text : tesserae.db.entities.Text
        Text whose feature frequencies are to be computed

    Returns
    -------
    dict [int, int]
        the key should be a feature index of type "form"; the associated
        value is the number of tokens in the text sharing at least one same
        feature type with the key word
    """
    tindex2mtindex = {}
    findex2mfindex = {}
    word_counts = Counter()
    word_feature_pairs = set()
    unit_proj = {
        '_id': False,
        'tokens.features.form': True
    }
    if feature != 'form':
        unit_proj['tokens.features.'+feature] = True
    db_cursor = connection.connection[Unit.collection].find(
        {'text': text.id, 'unit_type': 'line'},
        unit_proj
    )
    for unit in db_cursor:
        for token in unit['tokens']:
            cur_features = token['features']
            # use the form index as an identifier for this token's word
            # type
            cur_tindex = cur_features['form'][0]
            if cur_tindex not in tindex2mtindex:
                tindex2mtindex[cur_tindex] = len(tindex2mtindex)
            mtindex = tindex2mtindex[cur_tindex]
            # we want to count word types by matrix indices for faster
            # lookup when we get to the stage of counting up word type
            # occurrences
            word_counts[mtindex] += 1
            for cur_findex in cur_features[feature]:
                if cur_findex not in findex2mfindex:
                    findex2mfindex[cur_findex] = len(findex2mfindex)
                mfindex = findex2mfindex[cur_findex]
                # record when a word type is associated with a feature type
                word_feature_pairs.add((mtindex, mfindex))
    csr_rows = []
    csr_cols = []
    for mtindex, mfindex in word_feature_pairs:
        csr_rows.append(mtindex)
        csr_cols.append(mfindex)
    word_feature_matrix = csr_matrix(
        (
            np.ones(len(csr_rows), dtype=np.bool),
            (np.array(csr_rows), np.array(csr_cols))
        ),
        shape=(len(tindex2mtindex), len(findex2mfindex))
    )
    # if matching_words_matrix[i, j] == True, then the word represented by
    # position i shared at least one feature type with the word represented
    # by position j
    matching_words_matrix = word_feature_matrix.dot(
        word_feature_matrix.transpose())
    # since only matching tokens remain, the column indices indicate
    # which tokens match the token represented by row i; we need to
    # count up how many times each word appeared; first, we change match
    # indicators into the number of times the token indicated by the column was
    # found
    mtindices = []
    mtcounts = []
    for mtindex, count in word_counts.items():
        mtindices.append(mtindex)
        mtcounts.append(count)
    count_matrix = matching_words_matrix.dot(
        csr_matrix(
            (
                np.array(mtcounts),
                (np.array(mtindices), np.zeros(len(mtindices)))
            ),
            shape=(matching_words_matrix.shape[0], 1)
        )
    )
    # now, we can sum up each row to find the total number of times all
    # associated tokens appeared with the token represented by the row
    summed_rows = count_matrix.sum(axis=-1)
    # finally, we re-map matrix indices to feature indices
    mtindex2tindex = {
        mtindex: tindex for tindex, mtindex in tindex2mtindex.items()
    }
    return {
        mtindex2tindex[i]: freq for i, freq in enumerate(summed_rows.A1)
    }


def get_inverse_text_frequencies(connection, feature, text_id):
    """Get inverse frequency data (calculated by the given feature) for words
    in a particular text.

    This method assumes that for a given word type, the feature types
    extracted from any one instance of the word type will be the same as
    all other instances of the same word type.  Thus, further work would be
    necessary, for example, if features could be extracted based on word
    position.

    Parameters
    ----------
    connection : tesserae.db.mongodb.TessMongoConnection
    feature : str
        Feature category to be used in calculating frequencies
    text_id : bson.objectid.ObjectId
        ObjectId of the text whose feature frequencies are to be computed

    Returns
    -------
    dict [int, float]
        the key should be a feature index of type form; the associated
        value is the inverse of the average proportion of words in the text
        sharing at least one same feature type with the key word
    """
    tindex2mtindex = {}
    findex2mfindex = {}
    word_counts = Counter()
    word_feature_pairs = set()
    text_token_count = 0
    unit_proj = {
        '_id': False,
        'tokens.features.form': True
    }
    if feature != 'form':
        unit_proj['tokens.features.'+feature] = True
    db_cursor = connection.connection[Unit.collection].find(
        {'text': text_id, 'unit_type': 'line'},
        unit_proj
    )
    for unit in db_cursor:
        text_token_count += len(unit['tokens'])
        for token in unit['tokens']:
            cur_features = token['features']
            # use the form index as an identifier 
            # for this token's word type
            cur_tindex = cur_features['form'][0]
            units.append(cur_tindex)
            if cur_tindex not in tindex2mtindex:
                tindex2mtindex[cur_tindex] = len(tindex2mtindex)
            # mtindex is esseentially a counter
            # for the number of unique forms seen
            mtindex = tindex2mtindex[cur_tindex]
            # we want to count word types by matrix indices for faster
            # lookup when we get to the stage of counting up word type
            # occurrences
            word_counts[mtindex] += 1
            for cur_findex in cur_features[feature]:
                if cur_findex not in findex2mfindex:
                    findex2mfindex[cur_findex] = len(findex2mfindex)
                mfindex = findex2mfindex[cur_findex]
                # record when a word type is associated with a feature type
                word_feature_pairs.add((mtindex, mfindex))
    csr_rows = []
    csr_cols = []
    for mtindex, mfindex in word_feature_pairs:
        csr_rows.append(mtindex)
        csr_cols.append(mfindex)
    word_feature_matrix = csr_matrix(
        (
            np.ones(len(csr_rows), dtype=np.bool),
            (np.array(csr_rows), np.array(csr_cols))
        ),
        shape=(len(tindex2mtindex), len(findex2mfindex))
    )
    # if matching_words_matrix[i, j] == True, then the word represented by
    # position i shared at least one feature type with the word represented
    # by position j
    matching_words_matrix = word_feature_matrix.dot(
        word_feature_matrix.transpose())
    # since only matching tokens remain, the column indices indicate
    # which tokens match the token represented by row i; we need to
    # count up how many times each word appeared; first, we change match
    # indicators into the number of times the token indicated by the column was
    # found
    mtindices = []
    mtcounts = []
    for mtindex, count in word_counts.items():
        mtindices.append(mtindex)
        mtcounts.append(count)
    count_matrix = matching_words_matrix.dot(
        csr_matrix(
            (
                np.array(mtcounts),
                (np.array(mtindices), np.zeros(len(mtindices)))
            ),
            shape=(matching_words_matrix.shape[0], 1)
        )
    )
    # now, we can sum up each row to find the total number of times all
    # associated tokens appeared with the token represented by the row
    summed_rows = count_matrix.sum(axis=-1)
    # dividing total number of tokens by the sums gives us the inverse
    # frequencies
    sparse_freqs = text_token_count / summed_rows
    # finally, we re-map matrix indices to feature indices
    mtindex2tindex = {
        mtindex: tindex for tindex, mtindex in tindex2mtindex.items()
    }
    return {
        mtindex2tindex[i]: freq for i, freq in enumerate(sparse_freqs.A1)
    }
    
    
def get_sound_inverse_text_freq(connection, text_id):
    """Get the inverse frequencies of all the trigrams AKA sound features
    in a particular text.
    
    Formula:
    inv freq = 1/(occurences of trigram in text/total trigrams in text)

    Parameters
    ----------
    connection : tesserae.db.mongodb.TessMongoConnection
    text_id : bson.objectid.ObjectId
        ObjectId of the text whose feature frequencies are to be computed

    Returns
    -------
    dict [int, float]
        the key should be a feature index of type form; the associated
        value is the inverse frequency of the trigram
    """
    units = []
    unit_proj = {
        '_id': False,
        'tokens.features.form': True
    }
    unit_proj['tokens.features.sound'] = True
    db_cursor = connection.connection[Unit.collection].find(
        {'text': text_id, 'unit_type': 'line'},
        unit_proj
    )
    for unit in db_cursor:
        for token in unit['tokens']:
            cur_features = token['features']
            # use the sound feature index as an identifier. 
            # sound feature does not need to stay connected to its word
            for cur_tindex in cur_features['sound']:
                # continually append units as each line is processed
                units.append(cur_tindex)
    # count number of times each feature member appears in text
    units_count = Counter(units)
    # Frequency is the number of times a word occurs in a text 
    # divided by the total number of words in that text
    frequencies = {}
    inv_frequencies = {}
    N_text = len(units)
    for sound in units_count:
        frequencies[sound] = units_count[sound]/N_text
        inv_frequencies[sound] = 1/frequencies[sound]
    return inv_frequencies


