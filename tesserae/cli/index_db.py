#!/usr/bin/env python3
"""Index the Tesserae database

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

from tesserae.db import TessMongoConnection


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae.cli.index_db',
        description='Create indexes on the Tesserae database')

    p.add_argument(
        'db_cred',
        type=str,
        help=('path to database credentials file (see index_db.py for '
              'details)'))

    return p.parse_args(args)


def main():
    args = parse_args()

    with open(args.db_cred) as ifh:
        db_cred = json.load(ifh)

    conn = TessMongoConnection(db_cred['host'],
                               db_cred['port'],
                               db_cred['user'],
                               db_cred['password'],
                               db=db_cred['database'])

    conn.create_indices()


if __name__ == '__main__':
    main()
