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
        return _featurize_for_test
    raise ValueError(
        f'Could not find a featurizer for {language} to extract features of '
        f'type {feature}')


def _featurize_for_test(tokens):
    result = []
    for i, token in enumerate(tokens):
        if i % 3 == 0:
            result.append(['a'])
        elif i % 3 == 1:
            result.append(['a', 'b'])
        else:
            result.append(['c'])
    return result
