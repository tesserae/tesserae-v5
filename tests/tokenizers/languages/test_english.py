from tesserae.tokenizers import EnglishTokenizer


def test_normalize(token_connection):
    eng = EnglishTokenizer(token_connection)

    raw_lines = [
        '<English Test 1> Will this work as expected?',
        '<English Test 2> Porcupine',
        '<English Test 3> Old McDonald had a farm',
    ]

    ref_tokens = [
        'will', 'this', 'work', 'as', 'expected',
        'porcupine',
        'old', 'mcdonald', 'had', 'a', 'farm',
    ]

    tokens, tags = eng.normalize(raw_lines)

    correct = map(lambda x: x[0] == x[1], zip(tokens, ref_tokens))

    if not all(correct):
        for raw, actual, ref in zip(correct, raw_tokens, tokens, ref_tokens):
            if actual != ref:
                print('{}->{} (should be "{}")'.format(raw, actual, ref))

    assert all(correct)

    for tag, line in zip(tags, raw_lines):
        correct_tag = line[:line.find('>') + 1]
        assert tag == correct_tag


def test_tokenize(token_connection):
    # TODO implement once BaseTokenizer.tokenize is finalized
    assert False
