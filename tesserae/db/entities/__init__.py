from .entity import Entity
from .frequency import Frequency
from .match import Match
from .text import Text
from .token import Token
from .unit import Unit
from .swlist import StopwordsList

entity_map = {}
entity_map[Frequency.collection] = Frequency
entity_map[Match.collection] = Match
entity_map[Text.collection] = Text
entity_map[Token.collection] = Token
entity_map[Unit.collection] = Unit
entity_map[StopwordsList.collection] = StopwordsList

__all__ = ['Entity', 'Frequency', 'Match', 'Text', 'Token', 'Unit']
