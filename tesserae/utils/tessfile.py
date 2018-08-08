"""Smart I/O for .tess files.
"""

import hashlib
import os
import string
import warnings

# TODO: Write function to validate .tess file


class MalformedTessFileError(Exception):
    """Raised when a .tess line tag is not correct."""
    def __init__(self, fname, line=-1):
        if line >= 0:
            msg = 'Malformed .tess file in {} at line {}'.format(fname, line)
        else:
            msg = 'Malormed .tess file name {}'.format(fname)
        super(MalformedTessFileError, self).__init__(msg)


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
    path : str
        Path to the .tess file.
    mode : str
        File open mode ('r', 'w', 'a', etc.)
    buffer : bool
        If True, load file contents into memory on-the-fly. Otherwise, load in
        contents on initialization.
    hash : str
        MD5 hash of the file.
    metadata : tesserae.db.Text
        Text metdata from the database.

    """
    def __init__(self, path, mode='r', buffer=True, validate=False,
                 metadata=None):
        self.path = path
        self.mode = mode
        self.buffer = buffer
        self.fname = os.path.basename(path)
        self.metadata = metadata

        if buffer:
            self.file = open(path, 'r')
        else:
            self.file = []
            with open(path, 'r') as f:
                for line in f.readlines():
                    self.file.append(line)

        self.__hash = None
        self.__len = None

        if validate:
            self.validate()

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError()

        if self.buffer:
            self.file.seek(0)
            for _ in range(index + 1):
                line = self.file.readline()
            return line
        else:
            return self.file[index]

    def __len__(self):
        if self.__len is None:
            self.__len = sum([1 for _ in self.readlines()])
        return self.__len

    @property
    def hash(self):
        """The MD5 hash of the .tess file"""
        if self.__hash is None:
            hashinator = hashlib.md5()
            for line in self.readlines():
                hashinator.update(line.encode('utf-8'))
            self.__hash = hashinator.hexdigest()
        return self.__hash

    def readlines(self, include_tag=True):
        """Iterate over the lines of the .tess file in order.

        Yields
        ------
        line : str
            One line of the .tess file.
        """
        if self.buffer:
            self.file.seek(0)
            for line in self.file.readlines():
                if not include_tag:
                    start = line.find('>') + 1 if not include_tag else 0
                    line = line[start:]
                yield line
        else:
            for line in self.file:
                if not include_tag:
                    start = line.find('>') + 1 if not include_tag else 0
                    line = line[start:]
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
            tokens = line.strip(string.whitespace).split()
            for token in tokens:
                yield token

    def validate(self):
        """Determine if this file is a valid .tess file.

        Raises
        ------
        MalformedTessFileError
            If a tag in the file does not contain the proper information.
        """
        name, ext = os.path.splitext(self.fname)

        # Ensure that the file has the .tess extension
        if ext != '.tess':
            msg = 'Bad filename {}. tess files must end in .tess'.format(
                                                                    self.fname)
            warnings.warn(msg, warning.UserWarning)

        # Get the author and title from the filename
        parts = name.split('.')
        author, title = parts[:2]

        if len(parts) > 2:
            major = int(parts[-1])
        else:
            major = 1

        minor = 1

        for i, line in enumerate(self.readlines()):
            line = line.strip(string.whitespace)
            if len(line) > 5:
                # Ensure that a line tage exists
                i += 1
                tag_end = line.find('>')
                if tag_end < 0:
                    msg = '{} may be malformed on line {}'.format(
                                                            self.fname, line)
                    warnings.warn(msg, UserWarning)

                tag = line[:tag_end + 1]
                parts = tag.split()
                tag_author, tag_title = parts[0][1:-1], parts[1][:-1]
                maj_min = parts[-1][:-1].split('.')
                if len(maj_min) == 1:
                    tag_maj = int(maj_min[0])
                    tag_min = 1
                else:
                    tag_maj = int(maj_min[0])
                    tag_min = int(maj_min[1])

                # Ensure the tag author and title match the filename
                if author.find(tag_author) < 0 or title.find(tag_title) < 0:
                    msg = '{} may be malformed on line {}'.format(
                                                            self.fname, line)
                    warnings.warn(msg, UserWarning)

                # Ensure that the major part number is incrementing correctly
                if int(tag_maj) not in [major, major + 1]:
                    msg = '{} may be malformed on line {}'.format(
                                                            self.fname, line)
                    warnings.warn(msg, UserWarning)

                # Ensure that the minor part number is incrementing corectly
                if tag_maj == major and tag_min != minor:
                    msg = '{} may be malformed on line {}'.format(
                                                            self.fname, line)
                    warnings.warn(msg, UserWarning)
                elif tag_maj == major + 1 and tag_min != 1:
                    msg = '{} may be malformed on line {}'.format(
                                                            self.fname, line)
                    warnings.warn(msg, UserWarning)

                if tag_maj == major:
                    minor += 1
                else:
                    major += 1
                    minor = 2
