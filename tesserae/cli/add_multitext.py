#!/usr/bin/env python3
"""Add multitext functionality to Tesserae

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
"""
import argparse
import json
import logging
import sys
import traceback

from tqdm import tqdm

from tesserae.db import TessMongoConnection, Text
from tesserae.db.entities.text import TextStatus


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae.cli.add_multitext',
        description='Add multitext functionality to Tesserae')

    p.add_argument(
        'db_cred',
        type=str,
        help=('path to database credentials file (see index_db.py for '
              'details)'))
    p.add_argument('--home', default=os.path.expanduser('~'),
       help=('path to consider as home path (important to change when'
             'using this script on system where multitext data is stored'
             'somewhere other than the calling user\'s home directory)'))

    default_lfn = 'add_multitext.log'
    p.add_argument(
        '--lfn',
        type=str,
        default=default_lfn,
        help=f'Log FileName: path to log file (default: {default_lfn})')

    default_level = 'DEBUG'
    p.add_argument('--log',
                   type=str,
                   default=default_level,
                   help=f'logging level (default: {default_level})')

    return p.parse_args(args)


def build_logger(logfilename, loglevel):
    logger = logging.getLogger('add_multitext')
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

    conn = TessMongoConnection(db_cred['host'],
                               db_cred['port'],
                               db_cred['user'],
                               db_cred['password'],
                               db=db_cred['database'])

    os.environ['HOME'] = args.home
    from tesserae.utils.multitext import register_bigrams, MULTITEXT_SEARCH
    texts = conn.find(Text.collection)
    for text in tqdm(texts):
        if needs_multitext_enabled(text):
            logger.info(f'Extracting bigrams: {text.author}\t{text.title}')
            try:
                register_bigrams(conn, text)
            except KeyboardInterrupt:
                logger.info('KeyboardInterrupt')
                sys.exit(1)
            # we want to catch all other errors and log them
            except:  # noqa: E722
                logger.exception(f'Failed: {text.author}\t{text.title}')
                logger.exception(traceback.format_exc())


def needs_multitext_enabled(text):
    text_multitext_statuses = [
        (search_type, status_type)
        for _, search_type, status_type, _ in text.iterate_ingestion_details()
        if search_type == MULTITEXT_SEARCH
    ]
    if not text_multitext_statuses:
        return True
    if (MULTITEXT_SEARCH, TextStatus.FAILED) in text_multitext_statuses:
        return True
    return False


if __name__ == '__main__':
    main()
