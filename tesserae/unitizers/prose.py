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

        phrases = line.split(';')

        for phrase in phrases:
            tokens = list(
                map(lambda x: get_token_info(x, text.metadata.language).type,
                    phrase.split()))

            unit['text'] = text.path
            unit['index'] = i
            unit['unit_type'] = 'line'
            unit['raw'] = line
            unit['tokens'] = tokens

            i += 1

            units.append(unit)

    return units
