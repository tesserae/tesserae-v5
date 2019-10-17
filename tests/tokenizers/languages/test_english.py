from test_base_tokenizer import TestBaseTokenizer
from tesserae.tokenizers import EnglishTokenizer


class TestEnglishTokenizer(TestBaseTokenizer):
    __test_class_ = EnglishTokenizer

    def test_normalize(self):
        eng = self.__test_class__()

        raw_tokens = [
            'aardvark',
            'Porcupine',
            'McDonald',
        ]

        ref_tokens = [
            'aardvark',
            'porcupine',
            'mcdonald',
        ]

        tokens = eng.normalize(raw_tokens)

        correct = map(lambda x: x[0] == x[1], zip(raw_tokens, ref_tokens))

        if not all(correct):
            for raw, actual, ref in zip(correct, raw_tokens, tokens, ref_tokens):
                if actual != ref:
                    print('{}->{} (should be "{}")'.format(raw, actual, ref))

        assert all(correct)

    def test_tokenize(self):
        # TODO implement once BaseTokenizer.tokenize is finalized
        assert False
