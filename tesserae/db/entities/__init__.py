from .entity import Entity
from .feature import Feature
from .match import Match
from .multiresult import MultiResult
from .search import Search
from .swlist import StopwordsList
from .text import Text
from .token import Token
from .unit import Unit

entity_map = {}
entity_map[Feature.collection] = Feature
entity_map[Match.collection] = Match
entity_map[MultiResult.collection] = MultiResult
entity_map[Search.collection] = Search
entity_map[StopwordsList.collection] = StopwordsList
entity_map[Text.collection] = Text
entity_map[Token.collection] = Token
entity_map[Unit.collection] = Unit

__all__ = ['Entity', 'Feature', 'Match', 'MultiResult', 'Search',
           'Text', 'Token', 'Unit']
