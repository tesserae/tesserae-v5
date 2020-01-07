import collections
import re
import unicodedata

from tesserae.db.entities import Entity, Feature, Token


def _get_db_features_by_type(conn, language, feature_types):
    """Get Feature entities from the database, sorted by their types

    Parameters
    ----------
    conn : TessMongoConnection
        The database to query for features
    language : str
        The language of the features to get from the database
    feature_types : iterable of str
        The feature types to sort by

    Returns
    -------
    dict[str, list of Features]
        Mapping between feature type and the Features of that type
    """
    all_features = conn.find(Feature.collection, language=language)
    result = {ft: [] for ft in feature_types}
    for f in all_features:
        ft = f.feature
        if ft in result:
            result[ft].append(f)
    return result


class BaseTokenizer(object):
    """Tokenizer with global operations.

    Attributes
    ----------
    tokens : list of tesserae.db.Token
        The tokens encountered by this tokenizer in the order they were
        encountered.
    frequencies : collections.Counter
        Key/value store of normalized token forms (key) and their reapective
        counts (value).

    Notes
    -----
    To create a tokenizer for a language not included in Tesserae, subclass
    BaseTokenizer and override the ``normalize`` and ``featurize`` methods with
    functionality specific to the new language.

    """

    def __init__(self, connection):
        self.connection = connection

        # This pattern is used over and over again
        self.word_characters = 'a-zA-Z'
        self.diacriticals = \
            '\u0313\u0314\u0301\u0342\u0300\u0301\u0308\u0345'

        self.split_pattern = \
            '( / )|([\\s]+)|([^\\w\\d' + self.diacriticals + ']+)'

    def featurize(self, tokens):
        raise NotImplementedError

    def normalize(self, raw, split=False):
        """Standardize token representation for further processing.

        The global version of this function removes whitespace, non-word
        characters, and digits from the lowercase form of each raw token.

        Parameters
        ----------
        raw : str or list of str
            The string(s) to convert. Whitespace will be removed to provide a
            list of tokens.

        Returns
        -------
        normalized : list
            The list of tokens in normalized form.
        """
        # If dealing with a list of strings, attempt to join the individual
        # string entries with spaces.
        if isinstance(raw, list):
            raw = ' '.join(raw)

        # Extract .tess file markup and remove from the normalized string
        tags = re.findall(r'([<][^>]+[>])', raw, flags=re.UNICODE)
        raw = re.sub(r'[<][^>]+[>]\s+', r'', raw, flags=re.UNICODE)

        # Remove what appear to be Tesserae line delimiters
        raw = re.sub(r'/', r' ', raw, flags=re.UNICODE)

        # Apply lowercase and NKFD normalization to the token string
        normalized = unicodedata.normalize('NFKD', raw).lower()

        # If requested, split based on the language's split pattern.
        if split:
            normalized = re.split(self.split_pattern, normalized,
                                  flags=re.UNICODE)

        return normalized, tags

    def tokenize(self, raw, text=None):
        """Normalize and featurize the words in a string.

        Tokens are comprised of the raw string, normalized form, and features
        related to the words under study. This computes all of the relevant
        data and tracks token frequencies in one shot.

        Parameters
        ----------
        raw : str or list of str
            The string(s) to process. If a list, assumes that the string
            members of the list have already been split as intended (e.g.
            list elements were split on whitespace).
        text : tesserae.Text, optional
            Text metadata for associating tokens and frequencies with a
            particular text.

        Returns
        -------
        tokens : list of tesserae.db.Token
            The token entities to insert into the database.
        tags : list of str
            Metadata about the source text for unit bookeeping.
        features list of tesserae.db.Feature
            Features associated with the tokens to be inserted into the
            database.
        """
        # Compute the normalized forms of the input tokens, splitting the
        # result based on a regex pattern and discarding None values.
        normalized, tags = self.normalize(raw)
        tags = [t[:-1].split()[-1] for t in tags]

        # Compute the display version of each token by stripping the metadata
        # tags and converting newlines to their symbolic form.
        raw = re.sub(r'[<][^>]+[>]\s+', r'', raw, flags=re.UNICODE)
        raw = re.sub(r'/', r' ', raw, flags=re.UNICODE)
        raw = re.sub(r'[\n]', r' / ', raw, flags=re.UNICODE)

        # Split the display form into independent strings for each token,
        # discarding any None values.
        display = re.split(self.split_pattern, raw, flags=re.UNICODE)
        display = [t for t in display if t]

        # Compute the language-specific features of each token and add the
        # normalized forms as additional results.
        featurized = self.featurize(normalized)
        featurized['form'] = normalized

        # Get the text id from the metadata if it was passed in
        try:
            text_id = text
        except AttributeError:
            text_id = None

        # Get the token language from the metadata if it was passed in
        try:
            language = text.language
        except AttributeError:
            language = None

        tokens = []

        # Convert all computed features into entities, discarding duplicates.
        db_features = _get_db_features_by_type(self.connection, language,
                featurized.keys())
        results = [create_features(db_features[ft], text_id, language, ft,
            featurized[ft]) for ft in featurized.keys()]

        for feature_list, feature in results:
            featurized[feature] = feature_list

        # Prep the token objects
        norm_i = 0

        try:
            punctuation = self.connection.find('features', feature='punctuation')[0]
        except IndexError:
            punctuation = Feature(feature='punctuation', token='', index=-1)

        for i, d in enumerate(display):
            if re.search('[' + self.word_characters + ']+', d, flags=re.UNICODE):
                features = {key: val[norm_i]
                            for key, val in featurized.items()}
                norm_i += 1
            elif re.search(r'^[\d]+$', d, flags=re.UNICODE):
                features = {key: punctuation if key == 'form' else [punctuation]
                            for key in featurized.keys()}
            else:
                features = None

            t = Token(text=text, index=i, display=d, features=features)
            tokens.append(t)

        features = set()
        for val in featurized.values():
            if isinstance(val[0], list):
                for v in val:
                    features.update(v)
            else:
                features.update(val)

        return tokens, tags, list(features)


def create_features(db_features, text, language, feature, feature_list):
    """Create feature entities and register frequencies.


    """
    if isinstance(text, Entity):
        text = text.id
    db_features = {f.token: f for f in db_features}


    out_features = []
    for f in feature_list:
        if isinstance(f, collections.Sequence) and not isinstance(f, str):
            if f[0][0] == '<':
                continue
            feature_group = []
            for sub_f in f:
                if sub_f in db_features:
                    sub_f = db_features[sub_f]
                    try:
                        sub_f.frequencies[str(text)] += 1
                    except KeyError:
                        sub_f.frequencies[str(text)] = 1
                    feature_group.append(sub_f)
                else:
                    sub_f = Feature(feature=feature, token=sub_f,
                                    language=language,
                                    index=len(db_features),
                                    frequencies={str(text): 1})
                    db_features[sub_f.token] = sub_f
                    feature_group.append(sub_f)
            out_features.append(feature_group)

        else:
            if f[0] == '<':
                continue
            if f in db_features:
                f = db_features[f]
                try:
                    f.frequencies[str(text)] += 1
                except KeyError:
                    f.frequencies[str(text)] = 1
                out_features.append(f)
            else:
                f = Feature(feature=feature, token=f, language=language,
                            index=len(db_features), frequencies={str(text): 1})
                db_features[f.token] = f
                out_features.append(f)

    return out_features, feature
