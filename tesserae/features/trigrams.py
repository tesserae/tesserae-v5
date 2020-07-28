import re


def trigrammify(tokens):
    pattern = re.compile(r'\w')
    token_grams = []
    for token in tokens:
        grams = []
        characters = []
        for a in token:
            char = pattern.match(a)
            if char is not None:
                char = char.group()
                characters.append(char)
        final = len(characters) - 1
        if len(characters) < 3:
            grams = []
        else:
            for a in range(final - 1):
                grams.append(characters[a] + characters[a + 1] +
                             characters[a + 2])
        token_grams.append(grams)
    return token_grams
