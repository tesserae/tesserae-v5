import re

from tesserae.db import convert_to_entity, Unit
from tesserae.tokenizers.tokenize import get_token_info


@convert_to_entity(Unit)
def split_line_units(text, start=0, end=None):
    """Split a poem into line units.

    Parameters
    ----------
    text : tesserae.utils.TessFile
        TessFile object to provide the text to split.

    Returns
    -------
    lines : list of tesserae.db.Unit
    """
    units = []
    for line, i in enumerate(text.readlines()):
        tag_idx = line.find('>')

        if tag_idx >= 0:
            line = line[tag_idx + 1:]

        unit = {}
        unit['text'] = text.path
        unit['index'] = i
        unit['unit_type'] = 'line'
        unit['raw'] = line
        tokens = list(
            map(lambda x: get_token_info(x, text.metadata.language).type,
                line.split()))
        unit['tokens'] = tokens

        units.append(unit)

    return units


@convert_to_entity(Unit)
def split_phrase_units(text, start=0, end=None):
    """Split a poem into phrase units.

    Parameters
    ----------
    text : tesserae.utils.TessFile
        TessFile object to provide the text to split.

    Returns
    -------
    phrases : list of tesserae.db.Unit
    """
    units = []
    partial = False
    i = 0
    for line in text.readlines():
        unit = units.pop() if partial else {}

        tag_idx = line.find('>')

        if tag_idx >= 0:
            line = line[tag_idx + 1:]

        tokens = list(
            map(lambda x: get_token_info(x, text.metadata.language).type,
                line.split()))

        if not partial:
            unit['text'] = text.path
            unit['index'] = i
            unit['unit_type'] = 'line'
            unit['raw'] = line
            unit['tokens'] = tokens
        else:
            unit['raw'] = ' \\ '.join([unit['raw'], line])
            unit['tokens'].extend(tokens)

        units.append(unit)

        if re.match(r'[.;?]["]*$', line) is None:
            partial = True
        else:
            partial = False
            i += 1

    return units
