"""Add a new feature to already ingested texts"""
import argparse
import getpass
import logging
import sys
import traceback

from tqdm import tqdm

from tesserae.db import TessMongoConnection, Text
from tesserae.utils.ingest import add_feature


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae.cli.mass_add_feature',
        description='Add new feature to already ingested texts')

    db = p.add_argument_group(title='database',
                              description='database connection details')

    p.add_argument('feature', type=str, help='Name of feature to add')

    db.add_argument('--user', type=str, help='user to access the database as')
    db.add_argument('--password',
                    action='store_true',
                    help='pass to be prompted for a database password')
    db.add_argument('--host',
                    type=str,
                    default='127.0.0.1',
                    help='the host name or IP address of the MongoDB database')
    db.add_argument('--port',
                    type=str,
                    default=27017,
                    help='the port that the database listens on')
    db.add_argument('--database',
                    type=str,
                    default='tesserae',
                    help='the name of the database to access')
    return p.parse_args()


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
    """Ingest a text into Tesserae.

    Takes a .tess files and computes tokens, features, frequencies, and units.
    All computed components are inserted into the database.
    """
    args = parse_args()
    if args.password:
        password = getpass.getpass(prompt='Tesserae MongoDB Password: ')
    else:
        password = None

    logger = build_logger('mass_add_feature.log', 'DEBUG')
    connection = TessMongoConnection(args.host,
                                     args.port,
                                     args.user,
                                     password,
                                     db=args.database)
    for text in tqdm(connection.find(Text.collection)):
        logger.info(f'Adding feature "{args.feature}": '
                    f'{text.author}\t{text.title}')
        try:
            add_feature(connection, text, args.feature)
        except KeyboardInterrupt:
            logger.info('KeyboardInterrupt')
            sys.exit(1)
        # we want to catch all other errors and log them
        except:  # noqa: E722
            logger.exception(f'Failed to add feature "{args.feature}": '
                             f'{text.author}\t{text.title}')
            logger.exception(traceback.format_exc())


if __name__ == '__main__':
    main()
