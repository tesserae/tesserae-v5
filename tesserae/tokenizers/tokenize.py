import re

from cltk.corpus.utils.formatter import cltk_normalize
from cltk.semantics.latin.lookup import Lemmata
from cltk.stem.latin.j_v import JVReplacer

from tesserae.db.entities import convert_to_entity, Token
from tesserae.text_access.storage import retrieve_text_list


class InvalidLanguageError(Exception):
    def __init__(self, language):
        msg = '{} is not a valid Tesserae language'.format(language)
        super(InvalidLanguageError, self).__init__(msg)


def get_token_info(token, language, return_features=True, normalizer=None, featurizer=None):
    """Normalize and get lemmata, semantic, and sound data for a list of tokens

    Parameters
    ----------
    text : str or tesserae.db.Text
        The CTS URN of the text to load or a Text database entity.
    language : str
        The language that the tokens belong to.

    Other Parameters
    ----------------
    normalizer : callable, optional
        Custom normalizer function.
    lemmatizer : semantics.latin.lookup.Lemmata
        Custom lemmatizer, must be a subclass of the CLTK semantics lemmatizer
        or define a `lookup` function.
    """
    # TODO: get semantic meaning
    # TODO: get sound triples

    # Fall back to the standard normalizer function if no custom function was
    # provided.
    if normalizer is None:
        nfunc = '{}_normalizer'.format(language)
        normalizer = globals()[nfunc]

    # Fall back to the standard featurizer function if no custom function was
    # provided.
    if featurizer is None:
        ffunc = '{}_featurizer'.format(language)
        featurizer = globals()[ffunc]

    # Remove non-alphabetic characters
    token = re.sub(r'^\W+|\W+$', '', token.lower())
    token = re.sub(r'\d+', '', token)

    # Normalize for the selected language
    token_type = normalizer(token)

    # Get features for the token based on the language
    token_features = featurizer(token_type) if return_features else {}

    token = Token(language=language,
                  raw=token,
                  token_type=token_type,
                  **token_features)

    return token


def greek_normalizer(raw):
    """Normalize a single Greek word.

    Parameters
    ----------
    raw : str
        The word to normalize.

    Returns
    -------
    normalized : str
        The normalized string.
    """
    return cltk_normalize(raw.lower())


def greek_featurizer(token):
    """Get the features for a single Greek token.

    Parameters
    ----------
    token : str
        The token to featurize.

    Returns
    -------
    features : dict
        The features for the token.

    Notes
    -----
    Input should be sanitized with `greek_normalizer` prior to using this
    function.
    """
    features = {}
    features['lemmata'] = Lemmata('lemmata', 'greek').lookup(token)[0][1]
    return features


def latin_normalizer(raw):
    """Normalize a single Latin word.

    Parameters
    ----------
    raw : str
        The word to normalize.

    Returns
    -------
    normalized : str
        The normalized string.
    """
    return JVReplacer().replace(raw.lower())


def latin_featurizer(token):
    """Get the features for a single latin token.

    Parameters
    ----------
    token : str
        The token to featurize.

    Returns
    -------
    features : dict
        The features for the token.

    Notes
    -----
    Input should be sanitized with `latin_normalizer` prior to using this
    function.
    """
    features = {}
    features['lemmata'] = Lemmata('lemmata', 'latin').lookup(token)[0][1]
    return features
