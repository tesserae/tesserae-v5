import pytest

from tesserae.utils import TessFile
from tesserae.utils.tessfile import MalformedTessFileError

import hashlib
import io
import os
import random
import string


@pytest.fixture(scope='module')
def tessfile_list(tessfiles):
    tessfile_list = []
    for root, dirs, files in os.walk(tessfiles):
        if len(files) > 0 and root.find('new') < 0:
            tessfile_list.extend([os.path.join(root, f) for f in files])

    return tessfile_list


class TestTessFile(object):
    def test_init(self, tessfile_list):
        for f in tessfile_list:
            # Test initializing as buffer
            t = TessFile(f)
            assert t.path == f
            assert t.mode == 'r'
            assert t.buffer
            assert t._TessFile__hash is None
            assert t._TessFile__len is None
            assert isinstance(t.file, io.TextIOWrapper)
            assert t.file.name == f
            assert t.file.mode == 'r'

            # Test initializing with read
            result = []
            with open(f, 'r') as tess:
                for line in tess.readlines():
                    result.append(line)
            t = TessFile(f, buffer=False)
            assert t.path == f
            assert t.mode == 'r'
            assert not t.buffer
            assert t._TessFile__hash is None
            assert t._TessFile__len is None
            assert t.file == result

            # # Test initializing as buffer with validation
            # t = TessFile(f, validate=True)
            # assert t.mode == 'r'
            # assert t.buffer
            # assert t._TessFile__hash is None
            # assert t._TessFile__len is None
            # assert isinstance(t.file, io.TextIOWrapper)
            # assert t.file.name == f
            # assert t.file.mode == 'r'
            #
            # # Test initializing with read and validation
            # t = TessFile(f, buffer=False, validate=True)
            # assert t.mode == 'r'
            # assert not t.buffer
            # assert t._TessFile__hash is None
            # assert t._TessFile__len is None
            # assert t.file == result

        # Test instantiating with a non-existent file
        with pytest.raises(FileNotFoundError):
            t = TessFile('/foo/bar.tess')

        with pytest.raises(FileNotFoundError):
            t = TessFile('/foo/bar.tess', buffer=False)

        # Test instantiating with a directory as path
        with pytest.raises(IsADirectoryError):
            t = TessFile(os.path.dirname(os.path.abspath(__file__)))

        with pytest.raises(IsADirectoryError):
            t = TessFile(os.path.dirname(os.path.abspath(__file__)),
                         buffer=False)

    def test_getitem(self, tessfile_list):
        for f in tessfile_list:
            lines = []
            with open(f, 'r') as tess:
                for line in tess.readlines():
                    lines.append(line)

            indices = [i for i in range(len(lines))]

            # Test __getitem__ with buffering in order
            t = TessFile(f)
            for i in indices:
                assert t[i] == lines[i]

            # Test __getitem__ with buffering in order
            t = TessFile(f, buffer=False)
            for i in indices:
                assert t[i] == lines[i]

            random.shuffle(indices)

            # Test __getitem__ with buffering in order
            t = TessFile(f)
            for i in indices:
                assert t[i] == lines[i]

            # Test __getitem__ with buffering in order
            t = TessFile(f, buffer=False)
            for i in indices:
                assert t[i] == lines[i]

    def test_hash(self, tessfile_list):
        for f in tessfile_list:
            hashitizer = hashlib.md5()

            with open(f, 'r') as tess:
                hashitizer.update(tess.read().encode('utf-8'))
            h = hashitizer.hexdigest()

            # Test that the hash is computed correctly
            t = TessFile(f)
            assert t._TessFile__hash is None
            assert t.hash == h
            assert t._TessFile__hash == h

    def test_readlines(self, tessfile_list):
        for f in tessfile_list:
            lines = []
            with open(f, 'r') as tess:
                for line in tess.readlines():
                    lines.append(line)

            # Ensure that readlines works with a buffer
            t = TessFile(f)
            for i, line in enumerate(t.readlines()):
                assert line == lines[i]

            # Ensure that the buffer resets on second call
            reset = False
            for i, line in enumerate(t.readlines()):
                assert line == lines[i]
                reset = True
            assert reset

            # Ensure that readlines works with initial read
            t = TessFile(f, buffer=False)
            for i, line in enumerate(t.readlines()):
                assert line == lines[i]

            # Ensure that the iterator resets on second call
            reset = False
            for i, line in enumerate(t.readlines()):
                assert line == lines[i]
                reset = True
            assert reset

    def test_read_tokens(self, tessfile_list):
        for f in tessfile_list:
            lines = []
            with open(f, 'r') as tess:
                for line in tess.readlines():
                    lines.append(line)

            t_b = TessFile(f)
            t_r = TessFile(f, buffer=False)

            # Ensure that tokens omit the tag when requested
            # Grab all tokens from the text
            tokens = []
            for line in lines:
                start = line.find('>')
                if start >= 0:
                    tokens.extend(
                        line[start + 1:].strip(string.whitespace).split())

            # Test with buffer
            for i, token in enumerate(t_b.read_tokens()):
                # print(token, tokens[i])
                assert token == tokens[i]

            # Ensure that the iterator resets
            reset = False
            for i, token in enumerate(t_b.read_tokens()):
                assert token == tokens[i]
                reset = True
            assert reset

            # Test with initial read
            for i, token in enumerate(t_r.read_tokens()):
                assert token == tokens[i]

            # Ensure that the iterator resets
            reset = False
            for i, token in enumerate(t_r.read_tokens()):
                assert token == tokens[i]
                reset = True
            assert reset

            # Ensure that tokens include the tag when requested
            # Lines now start before the tag
            tokens = []
            for line in lines:
                tokens.extend(line.strip().split())

            # Test with buffer
            for i, token in enumerate(t_b.read_tokens(include_tag=True)):
                print(token, tokens[i])
                assert token == tokens[i]

            # Ensure that the iterator resets
            reset = False
            for i, token in enumerate(t_b.read_tokens(include_tag=True)):
                assert token == tokens[i]
                reset = True
            assert reset

            # Test with initial read
            for i, token in enumerate(t_r.read_tokens(include_tag=True)):
                assert token == tokens[i]

            # Ensure that the iterator resets
            reset = False
            for i, token in enumerate(t_r.read_tokens(include_tag=True)):
                assert token == tokens[i]
                reset = True
            assert reset

    def test_validate(self, tessfile_list):
        for f in tessfile_list:
            pass
