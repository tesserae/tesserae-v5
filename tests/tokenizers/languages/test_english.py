import re

from tesserae.tokenizers import EnglishTokenizer


def test_normalize(token_connection):
    eng = EnglishTokenizer(token_connection)

    raw_lines = (
        '<English Test 1> Will this work as expected?\n'
        '<English Test 2> Porcupine\n'
        '<English Test 3> Old McDonald had a farm\n'
    )

    ref_tokens = [
        'will', 'this', 'work', 'as', 'expected',
        'porcupine',
        'old', 'mcdonald', 'had', 'a', 'farm'
    ]

    tokens, tags = eng.normalize(raw_lines)

    correct = map(lambda x: x[0] == x[1], zip(tokens, ref_tokens))

    if not all(correct):
        for actual, ref in zip(tokens, ref_tokens):
            if actual != ref:
                print(f'"{actual}" should be "{ref}"')

    assert all(correct)

    for tag, line in zip(tags, raw_lines.split('\n')):
        correct_tag = line[:line.find('>') + 1]
        assert tag == correct_tag


def test_tokenize(token_connection):
    eng = EnglishTokenizer(token_connection)

    raw_lines = (
        '<English Test 1> Will this work as expected?\n'
        '<English Test 2> Porcupine\n'
        '<English Test 3> Old McDonald had a farm\n'
    )
    display_tokens = [
        'Will', 'this', 'work', 'as', 'expected',
        'Porcupine',
        'Old', 'McDonald', 'had', 'a', 'farm',
    ]
    form_tokens = [
        'will', 'this', 'work', 'as', 'expected',
        'porcupine',
        'old', 'mcdonald', 'had', 'a', 'farm',
    ]
    lemmata_tokens = [
        ['will'], ['this'], ['work'], ['a', 'as'], ['expect', 'expected'],
        ['porcupine'],
        ['old'], ['mcdonald'], ['have'], ['a'], ['farm']
    ]


    tokens, tags, features = eng.tokenize(raw_lines)
    tokens = [t for t in tokens if re.search(r'\w+', t.display)]

    correct = map(
        lambda x: x[0].display == x[1] and x[0].features['form'].token == x[2] and all([
            any(
                map(lambda y: lemma.token == y, x[3])
                )
            for lemma in x[0].features['lemmata']]),
        zip(tokens, display_tokens, form_tokens, lemmata_tokens))
    if not all(correct):
        for token, ref_display, ref_form, ref_lemmata in zip(
                tokens, display_tokens, form_tokens, lemmata_tokens):
            if token.display != ref_display:
                print(f'Display: "{token.display}" should be "{ref_display}"')
            if token.features['form'].token != ref_form:
                print('Form: "{}" should be "{}"'.format(
                    token.features['form'].token, ref_form))
            if not all([
                any(
                    map(lambda y: lemma.token == y, ref_lemmata)
                    )
                for lemma in token.features['lemmata']]):
                print('Lemmata: "{}" should be "{}"'.format(
                    [lemma.token for lemma in token.features['lemmata']], ref_lemmata))
    assert all(correct)

    for tag, line in zip(tags, raw_lines.split('\n')):
        correct_tag = line[:line.find('>')].split()[-1]
        assert tag == correct_tag
