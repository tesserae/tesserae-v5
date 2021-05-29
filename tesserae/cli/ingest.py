#!/usr/bin/env python3
"""Ingest a text into Tesserae.

Takes a .tess files and computes tokens, features, frequencies, and units. All
computed components are inserted into the database.
"""

import argparse
import getpass

from tesserae.db import TessMongoConnection, Text
from tesserae.tokenizers import tokenizer_map
from tesserae.utils.ingest import ingest_text


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae.cli.ingest',
        description='Ingest a text into the Tesserae database.')

    db = p.add_argument_group(title='database',
                              description='database connection details')
    text = p.add_argument_group(title='text', description='text metadata')

    p.add_argument('input', type=str, help='path to the .tess file to ingest')

    db.add_argument('--user', type=str, help='user to access the database as')
    db.add_argument('--password',
                    action='store_true',
                    help='pass to be prompted for a database password')
    db.add_argument('--host',
                    type=str,
                    default='127.0.0.1',
                    help='the host name or IP address of the MongoDB database')
    db.add_argument('--port',
                    type=int,
                    default=27017,
                    help='the port that the database listens on')
    db.add_argument('--database',
                    type=str,
                    default='tesserae',
                    help='the name of the database to access')

    text.add_argument('title', type=str, help='title of the text')
    text.add_argument('author', type=str, help='author of the text')
    text.add_argument('language',
                      type=str,
                      choices=list(tokenizer_map.keys()),
                      help='language the text was written in')
    text.add_argument('--year', type=int, help='year of authorship')
    text.add_argument('--prose',
                      action='store_true',
                      help=('pass to indicate that the text is prose, '
                            'otherwise it is considered poetry'))
    text.add_argument('--enable-multitext',
                      action='store_true',
                      help=('pass to compute and store multitext information '
                            'for this text'))

    return p.parse_args(args)


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

    connection = TessMongoConnection(args.host,
                                     args.port,
                                     args.user,
                                     password,
                                     db=args.database)

    text = Text(language=args.language,
                title=args.title,
                author=args.author,
                year=args.year,
                path=args.input,
                is_prose=args.prose)

    ingest_text(connection, text, enable_multitext=args.enable_multitext)


if __name__ == '__main__':
    main()
