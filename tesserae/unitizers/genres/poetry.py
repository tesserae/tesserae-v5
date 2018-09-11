import re

from tesserae.db import convert_to_entity, Unit
from tesserae.tokenizers.tokenize import get_token_info
from tesserae.unitizers.genres.base import BaseUnitizer


class PoetryUnitizer(BaseUnitizer):
    def unitize_lines(self, tokens, metadata):
        """Split a poem into line units.

        Parameters
        ----------
        tokens : list of tesserae.tokenizers.BaseTokenizer
            The tokens to group as units.
        metadata : tesserae.db.Text
            Text metadata for assigning the units to a text.

        Returns
        -------
        lines : list of tesserae.db.Unit

        Notes
        -----
        This method requires preprocessing the tokens with one of the
        tesserae tokenizers.

        See Also
        --------
        tesserae.tokenizers
        """
        units = []
        unit_idx = 0
        unit = Unit(text=metadata.path if metadata else None,
                    index=unit_idx,
                    unit_type='line')

        for t in tokens:
            if metadata:
                unit['text'] = metadata.path
            unit['index'] = unit_idx
            unit['unit_type'] = 'line'
            unit['tokens'].append(t.index)

            if re.search('[\n]| / ', t.raw, flags=re.UNICODE):
                units.append(unit)
                unit_idx += 1
                unit = Unit(text=metadata.path if metadata else None,
                            unit_type='line')

        return units

    def unitize_phrases(self, tokens, metadata):
        """Split a poem into line units.

        Parameters
        ----------
        tokens : list of tesserae.tokenizers.BaseTokenizer
            The tokens to group as units.
        metadata : tesserae.db.Text
            Text metadata for assigning the units to a text.

        Returns
        -------
        phrases : list of tesserae.db.Unit

        Notes
        -----
        This method requires preprocessing the tokens with one of the
        tesserae tokenizers.

        See Also
        --------
        tesserae.tokenizers
        """
        units = []
        unit = Unit(unit_type='phrase')
        unit_idx = 0

        for t in tokens:
            if metadata:
                unit['text'] = metadata.path
            unit['index'] = unit_idx
            unit['unit_type'] = 'line'
            unit['tokens'].append(t.index)

            if re.search('[\n]| / ', t.raw, flags=re.UNICODE):
                units.append(unit)
                unit = Unit(unit_type='phrase')
                unit_idx += 1

        return units
