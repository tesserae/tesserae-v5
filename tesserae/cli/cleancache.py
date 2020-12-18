#!/usr/bin/env python3
"""Ingest multiple texts into Tesserae.

One JSON file is required as input: the database credentials file

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

Optionally, logging options may be passed as command line arguments as well.
See the help message for more details.
"""
import argparse
import datetime
import logging
import json
import sys
import traceback

from tesserae.db import TessMongoConnection
from tesserae.db.entities import Search
from tesserae.utils.delete import remove_results


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae.cli.cleancache',
        description='Clean database of old results')

    p.add_argument(
        'db_cred',
        type=str,
        help=('path to database credentials file (see cleancache.py for '
              'details)'))

    default_lfn = 'clean.log'
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
    logger = logging.getLogger('cleancache')
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

    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=28)
    for_deletion = [
        Search.json_decode(s)
        for s in conn.connection[Search.collection].find(
            # https://stackoverflow.com/questions/11957595/mongodb-pymongo-query-with-datetime
            {'last_queried': {'$lt': cutoff}})]
    logger.info(
        'Number of Search entities out of date: {}'.format(len(for_deletion))
    )
    try:
        remove_results(conn, for_deletion)
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt')
        sys.exit(1)
    # we want to catch all other errors and log them
    except:  # noqa: E722
        logger.exception('Failed to delete out of date Search entities')
        logger.exception(traceback.format_exc())


if __name__ == '__main__':
    main()
