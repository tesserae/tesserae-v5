"""Class for specifying which units of a text to compare"""


class TextOptions:
    """A helper class for specifying which units of a text to compare

    Attributes
    ----------
    text : tesserae.db.entities.Text
        The text to include in the match
    unit_type : {'line', 'phrase'}
        The divisions of the text to use
    """

    def __init__(self, text, unit_type):
        self.text = text
        self.unit_type = unit_type
