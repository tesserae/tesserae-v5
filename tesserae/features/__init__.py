from . import trigrams


def get_featurizer(language, feature):
    """Get appropriate feature extraction callable

    Callable will take list[str] as input and give list[list[str]] as output,
    such that the list at position x in the output are the features extracted
    from the str at position x in the input.
    """
    if feature == 'sound':
        return trigrams.trigrammify
    elif feature == 'test':
        return trigrams.trigrammify
    raise ValueError(
        f'Could not find a featurizer for {language} to extract features of '
        f'type {feature}')
