#!/usr/bin/env python3
"""Look for a text in the Tesserae database"""

import argparse
import getpass
from pprint import pprint

from tesserae.db import TessMongoConnection, Text


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae.cli.find_text',
        description='Find a text in the Tesserae database.')

    db = p.add_argument_group(title='database',
                              description='database connection details')
    text = p.add_argument_group(title='text', description='text metadata')

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

    text.add_argument('--title', type=str, help='title of the text')
    text.add_argument('--author', type=str, help='author of the text')
    text.add_argument('--language',
                      type=str,
                      choices=['latin', 'greek'],
                      help='language the text was written in')

    return p.parse_args(args)


def main():
    """Look for a text in the Tesserae database"""
    args = parse_args()
    if args.password:
        password = getpass(prompt='Tesserae MongoDB Password: ')
    else:
        password = None

    connection = TessMongoConnection(args.host,
                                     args.port,
                                     args.user,
                                     password,
                                     db=args.database)

    kwargs = {}
    if args.title:
        kwargs['title'] = args.title
    if args.author:
        kwargs['author'] = args.author
    if args.language:
        kwargs['language'] = args.language
    pprint([t for t in connection.connection[Text.collection].find(kwargs)])


if __name__ == '__main__':
    main()
