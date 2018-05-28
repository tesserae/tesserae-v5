"""Smart I/O for .tess files.
"""


class TessFile(object):
    """Buffered/non-buffered reader for .tess file I/O.

    Parameters
    ----------
    path : str
        Path to the .tess file.
    mode : str
        File open mode ('r', 'w', 'a', etc.)
    buffer : bool
        If True, load file contents into memory on-the-fly. Otherwise, load in
        contents on initialization.

    Attributes
    ----------
    mode : str
        File open mode ('r', 'w', 'a', etc.)
    buffer : bool
        If True, load file contents into memory on-the-fly. Otherwise, load in
        contents on initialization.
    hash : str
        MD5 hash of the file.


    """
    def __init__(self, path, mode='r', buffer=True):
        self.mode = mode
        self.buffer = buffer

        if buffer:
            self.file = open(path, 'r')
        else:
            self.file = []
            with open(path, 'r') as f:
                for line in f.readlines():
                    self.file.append(line)

        self.__hash = None

    def __getitem__(self, index):
        # TODO: Retrieve specific line from ingested text
        if buffer:
            self.file.seek(0)
            for _ in range(index):
                line = self.file.readline()
            return line
        else:
            return self.file[index]

    @property
    def hash(self):
        """The MD5 hash of the .tess file"""
        if self.__hash is None:
            hashinator = hashlib.md5()
            for line in self.readlines():
                hashinator.update(line)
            self.__hash = hashinator.hexdigest()
        return self.__hash

    def readlines(self):
        """Iterate over the lines of the .tess file in order.

        Yields
        ------
        line : str
            One line of the .tess file.
        """
        if buffer:
            self.file.seek(0)
            for line in self.file.readlines():
                yield line
        else:
            for line in self.file:
                yield line

    def read_tokens(self, include_tag=False):
        """Iterate over the tokens of a .tess file in order.

        Parameters
        ----------
        include_tag : bool
            If True, include the starting tag with each line of the .tess file.
            Otherwise, only return the raw tokens.

        Yields
        ------
        token : str
            One token of the .tess file.
        """
        for line in self.readlines(include_tag=include_tag):
            start = line.find('>') if not include_tag else 0
            line = line[start:]
            tokens = line.split()
            for token in tokens:
                yield token
