"""For retrieving search results"""
from tesserae.db.entities import Feature, Match, MatchSet, Unit, Text


class MatchResult:
    """Template for organizing Match information

    See the API documentation for details on what each attribute is supposed to
    mean
    """

    def __init__(self, source, target, match_features, score, source_raw,
            target_raw, highlight):
        self.source = source
        self.target = target
        self.match_features = match_features
        self.score = score
        self.source_raw = source_raw
        self.target_raw = target_raw
        self.highlight = highlight

    def get_json_serializable(self):
        return {k: v for k, v in self.__dict__.items()}


def _get_display_tag(unit, text_cache):
    """Construct a display tag based on the Unit

    Parameters
    ----------
    unit : tesserae.db.entities.Unit
    text_cache : dict [ObjectId, str]
        Cache for text information in display tag
    """
    tag_parts = []
    tag_parts.append(text_cache[unit.text])
    if unit.tags:
        tag_parts.append(unit.tags[0])
    return ' '.join(tag_parts)


def _get_display_features(feature_ids, feature_cache):
    """Construct display features by ObjectId

    Parameters
    ----------
    feature_ids : list of bson.objectid.ObjectId
        List of ObjectIds for Tokens whose display information is wanted
    text_cache : dict [ObjectId, str]
        Cache for display features
    """
    return [feature_cache[f_id] for f_id in feature_ids]


def _gen_units(conn, db_matches, pos):
    found_units = conn.find(Unit.collection,
            _id=[db_m.units[pos] for db_m in db_matches])
    units_cache = {}
    for unit in found_units:
        units_cache[unit.id] = unit
    for db_m in db_matches:
        yield units_cache[db_m.units[pos]]


def _gen_source_units(conn, db_matches):
    for x in _gen_units(conn, db_matches, 0):
        yield x


def _gen_target_units(conn, db_matches):
    for x in _gen_units(conn, db_matches, 1):
        yield x


def _create_text_cache(conn, db_match_set):
    text_cache = {}
    texts = conn.find(Text.collection, _id=[text_id
        for text_id in db_match_set.texts])
    for text in texts:
        tmp = []
        if text.author:
            tmp.append(text.author)
        if text.title:
            tmp.append(text.title)
        text_cache[text.id] = ' '.join(tmp)
    return text_cache


def _create_feature_cache(conn, db_matches):
    feature_id_set = {f_id for db_m in db_matches for f_id in db_m.tokens}
    features_found = conn.find(Feature.collection,
            _id=[f_id for f_id in feature_id_set])
    return {f.id: f.token for f in features_found}


def get_results(connection, match_set_id):
    """Retrive results with associated MatchSet

    Parameters
    ----------
    match_set_id : bson.objectid.ObjectId
        ObjectId for MatchSet whose results you are trying to retrieve

    Returns
    -------
    list of MatchResult
    """
    result = []
    db_match_set = connection.find(MatchSet.collection, _id=match_set_id)[0]
    db_matches = connection.find(Match.collection, _id=db_match_set.matches)
    text_cache = _create_text_cache(connection, db_match_set)
    feature_cache = _create_feature_cache(connection, db_matches)
    for db_m, source_unit, target_unit in zip(db_matches,
            _gen_source_units(connection, db_matches),
            _gen_target_units(connection, db_matches)):
        result.append(MatchResult(
            source=_get_display_tag(source_unit, text_cache),
            target=_get_display_tag(target_unit, text_cache),
            match_features=_get_display_features(db_m.tokens, feature_cache),
            score=db_m.score,
            # TODO figure out how to retrieve raw text
            source_raw='',
            target_raw='',
            # TODO figure out what needs to go into highlight
            highlight=[]
        ))
    return result
