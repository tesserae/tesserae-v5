import pytest

from tesserae.unitizer import Unitizer

import json
import os
import re

from tesserae.db import Text
from tesserae.tokenizers import GreekTokenizer, LatinTokenizer
from tesserae.utils import TessFile


@pytest.fixture(scope='module')
def units(tessfiles):
    data = []
    for root, dirs, files in os.walk(tessfiles):
        if 'new' not in root and re.search(r'poetry|prose', root):
            fdata = {}
            for fname in files:
                parts = fname.split('.')
                if '.tess' in fname:
                    metadata = Text(
                        title=parts[0],
                        author=parts[1],
                        language='greek' if 'grc' in root else 'latin',
                        path=os.path.join(root, fname))
                    fdata['metadata'] = metadata
                if '.json' in fname:
                    feature = parts[2]
                    with open(os.path.join(root, fname), 'r') as f:
                        fdata[feature] = json.load(f)
            data.append(fdata)
    return data


class TestUnitizer(object):
    def test_init(self):
        u = Unitizer()
        assert hasattr(u, 'lines')
        assert u.lines == []
        assert hasattr(u, 'phrases')
        assert u.phrases == []

    def test_clear(self):
        u = Unitizer()

        vals = list(range(0, 100))

        u.lines.extend(vals)
        u.clear()
        assert hasattr(u, 'lines')
        assert u.lines == []
        assert hasattr(u, 'phrases')
        assert u.phrases == []

        u.lines.extend(vals)
        u.phrases.extend(vals)
        u.clear()
        assert hasattr(u, 'lines')
        assert u.lines == []
        assert hasattr(u, 'phrases')
        assert u.phrases == []

        u.lines.extend(vals)
        u.phrases.extend(vals)
        u.clear()
        assert hasattr(u, 'lines')
        assert u.lines == []
        assert hasattr(u, 'phrases')
        assert u.phrases == []

        for i in [None, 'a', 1, 1.0, True, False, b'a', r'a']:
            u.lines = i
            u.clear()
            assert hasattr(u, 'lines')
            assert u.lines == []
            assert hasattr(u, 'phrases')
            assert u.phrases == []

            u.phrases = i
            u.clear()
            assert hasattr(u, 'lines')
            assert u.lines == []
            assert hasattr(u, 'phrases')
            assert u.phrases == []

            u.lines = i
            u.phrases = i
            u.clear()
            assert hasattr(u, 'lines')
            assert u.lines == []
            assert hasattr(u, 'phrases')
            assert u.phrases == []

    def test_unitize(self, units):
        for unit in units:
            u = Unitizer()
            metadata = unit['metadata']
            tess = TessFile(metadata.path, metadata=metadata)
            tokens = unit['tokens']
            lines = unit['lines']
            phrases = unit['phrases']

            if metadata.language == 'greek':
                tokenizer = GreekTokenizer()
            elif metadata.language == 'latin':
                tokenizer = LatinTokenizer()

            tokenizer.clear()

            for i, line in enumerate(tess.readlines(include_tag=False)):
                stop = (i == len(tess) - 1)
                u.unitize(line, metadata, tokenizer=tokenizer, stop=stop)

            print(metadata.path)

            assert len(u.lines) == len(lines)
            for i in range(len(lines)):
                line_tokens = \
                    [tokenizer.tokens[j].form for j in u.lines[i].tokens
                     if re.search(r'[\w\d]', tokenizer.tokens[j].display,
                                  flags=re.UNICODE) and
                        tokenizer.tokens[j].form]

                correct_tokens = \
                    [tokens[j]['FORM'] for j in lines[i]['TOKEN_ID']
                     if 'FORM' in tokens[j] and tokens[j]['FORM']]

                if line_tokens != correct_tokens:
                    print('Line {}'.format(i))
                    print(line_tokens)
                    print(correct_tokens)

                assert line_tokens == correct_tokens

            print(u.phrases[-1].tokens)
            assert len(u.phrases) == len(phrases)
            for i in range(len(u.phrases)):
                phrase_tokens = \
                    [tokenizer.tokens[j].form for j in u.phrases[i].tokens
                     if re.search(r'[\w\d]', tokenizer.tokens[j].display,
                                  flags=re.UNICODE) and
                        tokenizer.tokens[j].form]

                correct_tokens = \
                    [tokens[j]['FORM'] for j in phrases[i]['TOKEN_ID']
                     if 'FORM' in tokens[j] and tokens[j]['FORM']]

                if phrase_tokens != correct_tokens:
                    print('Phrase {}'.format(i))
                    phrase_tokens = \
                        [tokenizer.tokens[j].form for j in u.phrases[i - 1].tokens
                         if re.search(r'[\w]', tokenizer.tokens[j].display,
                                      flags=re.UNICODE) and
                            tokenizer.tokens[j].form]

                    correct_tokens = \
                        [tokens[j]['FORM'] for j in phrases[i - 1]['TOKEN_ID']
                         if 'FORM' in tokens[j]]
                    print(phrase_tokens)
                    print(correct_tokens)

                assert phrase_tokens == correct_tokens

            assert len(u.phrases) == len(phrases)

            u.clear()
            tokenizer.clear()
