#!/usr/bin/env python3
"""Ingest multiple texts into Tesserae.

Two JSON files are required as input: the database credentials file and the
ingestion file.

The database credentials file must contain a JSON object with the following
attributes and values:
    * "user": user to access the database as
    * "password": password to use in accessing the database
    * "host": the host name or IP address of the MongoDB database
    * "port": the port number that the database listens on
    * "database": the name of the database to access
NEVER COMMIT THE DATABASE CREDENTIALS FILE TO GIT!

An example database credentials file would contain the following JSON object:
{
    "user": "me",
    "password": "no_one_will_guess_this",
    "host": "127.0.0.1",
    "port": 27017,
    "database": "tesserae"
}

The ingestion file must contain a list of JSON objects representing texts to be
ingested.  Each JSON object should have the following attributes and values:
    * "title": title of the work
    * "author": author of the work
    * "language": language in which the work was written
    * "path": location of .tess file associated with this work
    * "year": year when work was published (negative numbers refer to BCE)

As an example, the ingestion file might begin as follows:
[
    {
        "title": "aeneid",
        "author": "vergil",
        "language": "latin",
        "path": "/location/of/tess/files/vergil.aeneid.tess",
        "year": -19
    },
    {
        "title": ...
        ...
    },
    ...
]

For all .tess files referenced in the ingestion file, tokens, features,
frequencies, and units are computed. All computed components are inserted into
the database.

Optionally, logging options may be passed as command line arguments as well.
See the help message for more details.
"""
import argparse
import json
import logging
import sys
import traceback

from tqdm import tqdm

from tesserae.db import TessMongoConnection, Text
from tesserae.utils.ingest import ingest_text


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae.cli.mass_ingest',
        description='Tesserae Mass Ingest: ingest lots of texts')

    p.add_argument(
        'db_cred',
        type=str,
        help=('path to database credentials file (see mass_ingest.py for '
              'details)'))
    p.add_argument(
        'ingest',
        type=str,
        help='path to ingest file (see mass_ingest.py for details)')

    default_lfn = 'mass_ingest.log'
    p.add_argument(
        '--lfn',
        type=str,
        default=default_lfn,
        help=f'Log FileName: path to log file (default: {default_lfn})')

    default_level = 'DEBUG'
    p.add_argument(
        '--log',
        type=str,
        default=default_level,
        help=f'logging level (default: {default_level})')

    return p.parse_args(args)


def build_logger(logfilename, loglevel):
    logger = logging.getLogger('mass_ingest')
    # https://docs.python.org/3/howto/logging.html#logging-to-a-file
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logger.setLevel(numeric_level)
    fh = logging.FileHandler(logfilename)
    fh.setLevel(loglevel)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def main():
    args = parse_args()
    logger = build_logger(args.lfn, args.log)

    with open(args.db_cred) as ifh:
        db_cred = json.load(ifh)

    conn = TessMongoConnection(
        db_cred['host'], db_cred['port'], db_cred['user'], db_cred['password'],
        db=db_cred['database']
    )

    with open(args.ingest) as ifh:
        texts = [Text.json_decode(t) for t in json.load(ifh)]

    for text in tqdm(texts):
        logger.info(f'Starting ingest: {text.author}\t{text.title}')
        try:
            ingest_text(conn, text)
        except KeyboardInterrupt:
            logger.info('KeyboardInterrupt')
            sys.exit(1)
        # we want to catch all other errors and log them
        except:  # noqa: E722
            logger.exception(f'Failed to ingest: {text.author}\t{text.title}')
            logger.exception(traceback.format_exc())


if __name__ == '__main__':
    main()
