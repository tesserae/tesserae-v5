import argparse
import getpass

from tesserae.db import TessMongoConnection


def parse_args(args=None):
    p = argparse.ArgumentParser(
        prog='tesserae_ingest',
        description='Ingest a text into the Tesserae database.')

    db = p.add_argument_group(
        title='database',
        description='database connection details')
    text = p.add_argument_group(
        title='text',
        description='text metadata')

    db.add_argument(
        '--user',
        type=str,
        help='user to access the database as')
    db.add_argument(
        '--password',
        action='store_true',
        help='pass to be prompted for a database password')
    db.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='the host name or IP address of the MongoDB database')
    db.add_argument(
        '--port',
        type=str,
        default=27017,
        help='the port that the database listens on')
    db.add_argument(
        '--database',
        type=str,
        default='tesserae',
        help='the name of the database to access')

    text.add_argument(
        '--search',
        type=str,
        help='database search ID to serialize')
    text.add_argument(
        '--format',
        choices=['csv', 'json', 'xml'],
        default='csv',
        help='output format')
    text.add_argument(
        '--path',
        type=str,
        help='path to write export to (standard out if not supplied)')
    text.add_argument(
        '--delimiter',
        type=str,
        default=',',
        help='CSV row item delimiting character')

    return p.parse_args(args)


def main(connection, search_id, format, filepath=None, delimiter=','):
    if format == 'json':
        from tesserae.utils.exports.json import dump, dumps
        args = (connection, search_id)
        if filepath:
            args = args + (filepath,)
    elif format == 'xml':
        from tesserae.utils.exports.xml import dump, dumps
        args = (connection, search_id)
        if filepath:
            args = args + (filepath,)
    else:
        from tesserae.utils.exports.csv import dump, dumps
        args = (connection, search_id)
        if filepath:
            args = args + (filepath,)
        args = args + (delimiter,)


    if filepath:
        dump(*args)
    else:
        print(dumps(*args))


if __name__ == '__main__':
    args = parse_args()
    if args.password:
        password = getpass.getpass(prompt='Tesserae MongoDB Password: ')
    else:
        password = None

    connection = TessMongoConnection(
        args.host, args.port, args.user, password, db=args.database)
    main(connection, args.search, args.format, filepath=args.path)