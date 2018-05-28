import pytest

from tesserae.utils import TessFile

import io
import os


@pytest.fixture(scope='module')
def tessfile_list(tessfiles):
    tessfile_list = []
    for root, dirs, files in os.walk(tessfiles):
        if len(files) > 0:
            tessfile_list.extend([os.path.join(root, f) for f in files])

    return tessfile_list


class TestTessFile(object):
    def test_init(self, tessfile_list):
        for f in tessfile_list:
            # Test initializing as buffer
            t = TessFile(f)
            assert t.mode == 'r'
            assert t.buffer
            assert isinstance(t.file, io.TextIOWrapper)
            assert t.file.name == f
            assert t.file.model == 'r'

            # Test initializing with read
            result = []
            with open(f, 'r') as tess:
                for line in tess.readlines():
                    result.append(line)
            t = TessFile(f, buffer=False)
            assert t.mode == 'r'
            assert not t.buffer
            assert t.file == result

    def test_getitem(self, tessfiles):
        pass

    def test_hash(self, tessfiles):
        pass

    def test_readlines(self, tessfiles):
        pass

    def test_read_tokens(self, tessfiles):
        pass
