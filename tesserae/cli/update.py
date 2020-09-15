#!/usr/bin/env python3
"""Update texts in the Tesserae database"""

import argparse
import getpass
import json

from tesserae.db import TessMongoConnection, Text


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae.cli.update',
        description='Update texts in the Tesserae database.')
    p.add_argument('json_updates',
                   type=str,
                   help=('path to JSON file containing updates; '
                         'the file should consist of a list of JSON objects, '
                         'where each object contains an "_id" field '
                         'associated with the string representation of the '
                         'database identifier of a text to update; all other '
                         'fields and associated values in the object will be '
                         'added or updated to that text'))

    db = p.add_argument_group(title='database',
                              description='database connection details')

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

    return p.parse_args(args)


def main():
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

    with open(args.json_updates, encoding='utf-8') as ifh:
        raw_updates = json.load(ifh)
    connection.update([Text.json_decode(t) for t in raw_updates])


if __name__ == '__main__':
    main()
