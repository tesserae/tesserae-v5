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
import logging
import json
import sys
import traceback

from tesserae.db import TessMongoConnection
from tesserae.db.entities import Feature
from tesserae.data import load_greek_to_latin


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae.cli.setupdb',
        description='Setup database for the first time')

    p.add_argument(
        'db_cred',
        type=str,
        help=('path to database credentials file (see cleancache.py for '
              'details)'))

    default_lfn = 'setupdb.log'
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
    logger = logging.getLogger('setupdb')
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


def register_features(conn, language, feature_type, candidates):
    """Registers features in the database

    Parameters
    ----------
    conn : TessMongoConnection
    language : str
        Language of the Feature entities to insert into database
    feature_type : str
        Type of feature the Feature entities to be inserted into the database
        are supposed to be
    candidates : set[str]
        Instances of a feature type that might need to be added to the database
    """
    already_registered = {
        f.token
        for f in conn.find(Feature.collection, language=language,
                           feature=feature_type)
    }
    to_be_registered = [c for c in candidates if c not in already_registered]
    conn.insert([
        Feature(language=language, feature=feature_type, token=c, index=i)
        for c, i in zip(
            to_be_registered,
            range(len(already_registered),
                  len(already_registered) + len(to_be_registered)))
    ])


def register_greek_features(conn, feature_type, candidates):
    register_features(conn, 'greek', feature_type, candidates)


def register_latin_features(conn, feature_type, candidates):
    register_features(conn, 'latin', feature_type, candidates)


def register_greek_to_latin_lemmata(conn):
    """Update database with features from Greek to Latin data

    Parameters
    ----------
    conn : TessMongoConnection

    """
    g2l = load_greek_to_latin()
    register_greek_features(conn, 'lemmata', set(g2l.keys()))
    register_latin_features(conn, 'lemmata', set(
        w for translations in g2l.values() for w in translations))


def main():
    args = parse_args()
    logger = build_logger(args.lfn, args.log)

    with open(args.db_cred) as ifh:
        db_cred = json.load(ifh)

    conn = TessMongoConnection(
        db_cred['host'], db_cred['port'], db_cred['user'], db_cred['password'],
        db=db_cred['database']
    )
    try:
        logger.info('Indexing database')
        conn.create_indices()
        logger.info('Registering Greek to Latin Lemmata')
        register_greek_to_latin_lemmata(conn)
    except KeyboardInterrupt:
        logger.info('KeyboardInterrupt')
        sys.exit(1)
    # we want to catch all other errors and log them
    except:  # noqa: E722
        logger.exception('Failed initial set up of database')
        logger.exception(traceback.format_exc())


if __name__ == '__main__':
    main()
