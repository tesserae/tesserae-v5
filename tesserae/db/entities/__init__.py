from .entity import Entity
from .feature_set import FeatureSet
from .frequency import Frequency
from .match import Match
from .match_set import MatchSet
from .text import Text
from .unit import Unit
from .token import Token
from .swlist import StopwordsList

entity_map = {}
entity_map[FeatureSet.collection] = FeatureSet
entity_map[Frequency.collection] = Frequency
entity_map[Match.collection] = Match
entity_map[MatchSet.collection] = MatchSet
entity_map[Text.collection] = Text
entity_map[Token.collection] = Token
entity_map[Unit.collection] = Unit
entity_map[StopwordsList.collection] = StopwordsList

__all__ = ['Entity', 'FeatureSet', 'Frequency', 'Match', 'MatchSet', 'Text',
           'Token', 'Unit']
