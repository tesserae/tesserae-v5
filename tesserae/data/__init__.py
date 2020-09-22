import gzip
import pickle
import pkg_resources


def _load_data_dictionary(filename):
    return pickle.loads(
        gzip.decompress(pkg_resources.resource_string(__name__, filename)))


def load_greek_to_latin():
    """Retrieves Greek to Latin translation dictionary

    Returns
    -------
    dict[str, Any]
        Maps a Greek lemma to a list of Latin synonyms; these Latin synonyms
        are Latin lemmata
    """
    return _load_data_dictionary('g_l.pickle.gz')


def load_synonym_dictionary(language, synonym_type):
    """Retrieves synonymy dictionary

    Returns
    -------
    dict[str, Any]
        Maps a Greek lemma to a list of Greek synonyms; these Greek synonyms
        are Greek lemmata
    """
    syn_type_to_code = {'semantic': 'syn', 'semantic + lemmata': 'syn_lem'}
    code = syn_type_to_code[synonym_type]
    return _load_data_dictionary(f'fixed_{language}_{code}.pickle.gz')
