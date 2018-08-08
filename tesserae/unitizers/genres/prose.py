import re

from tesserae.db import convert_to_entity, Unit
from tesserae.tokenizers.tokenize import get_token_info
from tesserae.unitizers.genres.base import BaseUnitizer


class ProseUnitizer(BaseUnitizer):
    @convert_to_entity(Unit)
    def unitize_lines(self, tessfile, tokenizer):
        """Split prose into line units.

        Parameters
        ----------
        text : tesserae.utils.TessFile
            TessFile object to provide the text to split.

        Returns
        -------
        lines : list of tesserae.db.Unit
        """
        units = []
        for i, line in enumerate(tessfile.readlines()):
            line = line.strip()
            print(i)
            if i == 756:
                print(i, line)
            if not line or not re.match(r'^[\w]+$', line, flags=re.UNICODE):
                continue

            tag_idx = line.find('>')

            if tag_idx >= 0:
                line = line[tag_idx + 1:]

            unit = {}
            unit['text'] = tessfile.path
            unit['index'] = i
            unit['unit_type'] = 'line'
            unit['raw'] = line
            tokens = tokenizer.normalize(line)
            unit['tokens'] = tokens

            units.append(unit)

        return units

    @convert_to_entity(Unit)
    def unitize_phrases(self, tessfile, tokenizer):
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
        for line in tessfile.readlines():
            tag_idx = line.find('>')

            if tag_idx >= 0:
                line = line[tag_idx + 1:]
            line = line.strip()

            phrases = re.split(r'([;.?:!])', line)

            for phrase in phrases[:-1]:
                unit = {}
                tokens = tokenizer.normalize(phrase)

                unit['text'] = tessfile.path
                unit['index'] = i
                unit['unit_type'] = 'phrase'
                unit['raw'] = phrase
                unit['tokens'] = tokens

                i += 1

                units.append(unit)

        return units


# @convert_to_entity(Unit)
# def split_phrase_units(text):
#     """Split prose into phrase units.
#
#     Parameters
#     ----------
#     text : tesserae.utils.TessFile
#         TessFile object to provide the text to split.
#
#     Returns
#     -------
#     phrases : list of tesserae.db.Unit
#     """
#     units = []
#     i = 0
#     for line in text.readlines():
#         tag_idx = line.find('>')
#
#         if tag_idx >= 0:
#             line = line[tag_idx + 1:]
#         line = line.strip()
#
#         phrases = re.split(r'([;.?])', line)
#
#         for phrase in phrases[:-1]:
#             if phrase in [';', '.', '?']:
#                 units[-1]['raw'] += phrase
#                 continue
#             elif re.match(r'^\w+$', phrase):
#                 continue
#             unit = {}
#             tokens = list(
#                 map(lambda x: get_token_info(x, text.metadata.language, return_features=False).token_type,
#                     phrase.split()))
#
#             unit['text'] = text.path
#             unit['index'] = i
#             unit['unit_type'] = 'phrase'
#             unit['raw'] = phrase
#             unit['tokens'] = tokens
#
#             i += 1
#
#             units.append(unit)
#
#     return units
