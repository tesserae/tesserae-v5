import gzip
import pickle
import pkg_resources


def _load_data_dictionary(filename):
    return pickle.loads(gzip.decompress(
        pkg_resources.resource_string(__name__, filename)))


def load_greek_to_latin():
    """Retrieves Greek to Latin translation dictionary

    Returns
    -------
    dict[str, Any]
        Maps Greek lemma to a tuple of Latin synonyms; these Latin synonyms are
        Latin lemmata
    """
    return _load_data_dictionary('g_l.pickle.gz')
