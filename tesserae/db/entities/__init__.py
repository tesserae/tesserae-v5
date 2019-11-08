from .entity import Entity
from .feature import Feature
from .feature_set import FeatureSet
from .frequency import Frequency
from .match import Match
from .match_set import MatchSet
from .results_pair import ResultsPair
from .swlist import StopwordsList
from .text import Text
from .token import Token
from .unit import Unit

entity_map = {}
entity_map[Feature.collection] = Feature
entity_map[FeatureSet.collection] = FeatureSet
entity_map[Frequency.collection] = Frequency
entity_map[Match.collection] = Match
entity_map[MatchSet.collection] = MatchSet
entity_map[ResultsPair.collection] = ResultsPair
entity_map[StopwordsList.collection] = StopwordsList
entity_map[Text.collection] = Text
entity_map[Token.collection] = Token
entity_map[Unit.collection] = Unit

__all__ = ['Entity', 'Feature', 'Frequency', 'Match', 'MatchSet', 'Text',
           'Token', 'Unit']
