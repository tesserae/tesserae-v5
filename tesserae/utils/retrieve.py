"""For retrieving search results"""
from tesserae.db.entities import Feature, Match, Unit, Text


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


def _get_display_tag(connection, unit_id, text_cache):
    """Construct a display tag based on the Unit

    Parameters
    ----------
    connection : tesserae.db.mongodb.TessMongoConnection
    unit_id : bson.objectid.ObjectId
        ObjectId for Unit whose display information is wanted
    text_cache : dict [ObjectId, str]
        Cache for text information in display tag
    """
    unit = connection.find(Unit.collection, _id=unit_id)[0]

    if unit.text not in text_cache:
        text = connection.find(Text.collection, _id=unit.text)[0]
        tmp = []
        if text.author:
            tmp.append(text.author)
        if text.title:
            tmp.append(text.title)
        text_cache[unit.text] = ' '.join(tmp)

    tag_parts = []
    tag_parts.append(text_cache[unit.text])
    if unit.tags:
        tag_parts.append(unit.tags[0])
    return ' '.join(tag_parts)


def _get_display_features(connection, feature_ids, feature_cache):
    """Construct display features by ObjectId

    Parameters
    ----------
    connection : tesserae.db.mongodb.TessMongoConnection
    feature_ids : list of bson.objectid.ObjectId
        List of ObjectIds for Tokens whose display information is wanted
    text_cache : dict [ObjectId, str]
        Cache for display features
    """
    not_in_cache = [f_id for f_id in feature_ids if f_id not in feature_cache]
    if not_in_cache:
        grabbed_from_db = connection.find(Feature.collection, _id=not_in_cache)
        for f in grabbed_from_db:
            feature_cache[f.id] = f.token
    return [feature_cache[f_id] for f_id in feature_ids]


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
    db_matches = connection.find(Match.collection, match_set=match_set_id)
    text_cache = {}
    feature_cache = {}
    for db_m in db_matches:
        result.append(MatchResult(
            source=_get_display_tag(connection, db_m.units[0], text_cache),
            target=_get_display_tag(connection, db_m.units[1], text_cache),
            match_features=_get_display_features(connection, db_m.tokens,
                feature_cache),
            score=db_m.score,
            # TODO figure out how to retrieve raw text
            source_raw='',
            target_raw='',
            # TODO figure out what needs to go into highlight
            highlight=[]
        ))
    return result
