#!/usr/bin/env python3
"""Re-try ingestion on texts

Two JSON files are required as input: the database credentials file and the
reingestion file.

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

The reingestion file is a tab delimited file where the first item of each line
is the name of an author and the second item is the title of a work.

As an example, the reingestion file might begin as follows:
vergil  aeneid
lucan   bellum civile
...

During reingestion, the author and title names will be used to find works
already recorded in the database as Text entities, eliminate these entities and
all other entities referencing them, and then ingest the works anew, creating
or updating tokens, features, frequencies, and units.

Optionally, logging options may be passed as command line arguments as well.
See the help message for more details.
"""
import argparse
import json
import logging
import sys

from tqdm import tqdm

from tesserae.db import TessMongoConnection, Text
from tesserae.utils.ingest import reingest_text


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae.cli.mass_reingest.py',
        description='Tesserae Mass Reingest: reingest lots of texts')

    p.add_argument(
        'db_cred',
        type=str,
        help=('path to database credentials file (see mass_reingest.py for '
            'details)'))
    p.add_argument(
        'reingest',
        type=str,
        help='path to reingest file (see mass_reingest.py for details)')

    default_lfn = 'mass_reingest.py.log'
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
    logger = logging.getLogger('mass_reingest')
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

    with open(args.reingest) as ifh:
        texts = []
        for line in ifh:
            line = line.strip()
            if line:
                items = line.split('\t')
                texts.append(Text(author=items[0], title=items[1]))

    texts = conn.aggregate(
        Text.collection,
        [
            {
                '$match': {'$or': [{
                    'author': t.author,
                    'title': t.title
                } for t in texts]}
            }
        ]
    )

    for text in tqdm(texts):
        logger.info(f'Starting reingest: {text.author}\t{text.title}')
        try:
            reingest_text(conn, text)
        except KeyboardInterrupt:
            logger.info('KeyboardInterrupt')
            sys.exit(1)
        except:
            logger.exception(
                f'Failed to reingest: {text.author}\t{text.title}')


if __name__ == '__main__':
    main()
