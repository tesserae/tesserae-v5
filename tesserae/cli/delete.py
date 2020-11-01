#!/usr/bin/env python3
"""Delete a text from Tesserae"""

import argparse
import getpass

from bson.objectid import ObjectId

from tesserae.db import TessMongoConnection, Text
from tesserae.utils.delete import remove_text


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae.cli.delete',
        description='Delete a text from the Tesserae database.')
    p.add_argument('text_id', type=str, help='ObjectId of text to delete')

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
    """Delete a text from Tesserae"""
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

    text_id = ObjectId(args.text_id)
    found = connection.find(Text.collection, _id=text_id)
    if not found:
        raise ValueError(f'Could not find text with ID {args.text_id}')
    remove_text(connection, found[0])


if __name__ == '__main__':
    main()
