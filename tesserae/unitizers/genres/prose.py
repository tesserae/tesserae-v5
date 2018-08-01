import re

from tesserae.db import convert_to_entity, Unit
from tesserae.tokenizers.tokenize import get_token_info


@convert_to_entity(Unit)
def split_phrase_units(text):
    """Split prose into phrase units.

    Parameters
    ----------
    text : tesserae.utils.TessFile
        TessFile object to provide the text to split.

    Returns
    -------
    phrases : list of tesserae.db.Unit
    """
    units = []
    i = 0
    for line in text.readlines():
        tag_idx = line.find('>')

        if tag_idx >= 0:
            line = line[tag_idx + 1:]
        line = line.strip()

        phrases = re.split(r'([;.?])', line)

        for phrase in phrases[:-1]:
            if phrase in [';', '.', '?']:
                units[-1]['raw'] += phrase
                continue
            elif re.match(r'^\w+$', phrase):
                continue
            unit = {}
            tokens = list(
                map(lambda x: get_token_info(x, text.metadata.language, return_features=False).token_type,
                    phrase.split()))

            unit['text'] = text.path
            unit['index'] = i
            unit['unit_type'] = 'phrase'
            unit['raw'] = phrase
            unit['tokens'] = tokens

            i += 1

            units.append(unit)

    return units
