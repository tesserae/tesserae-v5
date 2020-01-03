#!/usr/bin/env python3
"""Ingest a text into Tesserae.

Takes a .tess files and computes tokens, features, frequencies, and units. All
computed components are inserted into the database.
"""

import argparse
import getpass
import time
import uuid

from tesserae.db import TessMongoConnection
from tesserae.db.entities import Search
from tesserae.matchers.sparse_encoding import SparseMatrixSearch
from tesserae.utils.ingest import ingest_text


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

    db.add_argument('--user',
                    type=str,
                    default=None,
                    help='user to access the database as')
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

    text.add_argument('--source',
                      type=str,
                      help='title of the source text')
    text.add_argument('--target',
                      type=str,
                      help='title of the target text')
    text.add_argument('--unit',
                      type=str,
                      choices=['line', 'phrase'],
                      help='units to match on')
    text.add_argument('--feature',
                      type=str,
                      choices=['form', 'lemmata'],
                      help='feature to match on')
    text.add_argument('--n-stopwords', type=int, default=10,
                      help='size of the stoplist')
    text.add_argument('--stopword-basis', choices=['corpus', 'texts'],
                      default='corpus', help='basis for stopword frequencies')
    text.add_argument('--score-basis', choices=['word', 'stem'],
                      default='word', help='compute frequency by word or individual stem')
    text.add_argument('--frequency-basis', choices=['corpus', 'texts'],
                      default='corpus', help='')
    text.add_argument('--max-distance', type=int, default=10,
                      help='maximum allowable ditance between match tokens')
    text.add_argument('--distance-metric', choices=['span', 'frequency'],
                      default='span', help='')
    text.add_argument('--min-score', type=int, default=6,
                      help='size of the stoplist')
    text.add_argument('--parallel', action='store_true',
                      help='enable parallel processing during search')

    return p.parse_args(args)


def main():
    """Ingest a text into Tesserae.

    Takes a .tess files and computes tokens, features, frequencies, and units.
    All computed components are inserted into the database.
    """
    args = parse_args()
    if args.password:
        password = getpass(prompt='Tesserae MongoDB Password: ')
    else:
        password = None

    connection = TessMongoConnection(
        args.host, args.port, args.user, password, db=args.database)

    source = connection.find('texts', title=args.source.lower())[0]
    target = connection.find('texts', title=args.target.lower())[0]

    engine = SparseMatrixSearch(connection)
    start = time.time()
    search = Search(results_id=uuid.uuid4().hex)
    connection.insert(search)
    texts, params, matches = engine.match(search,
                  [source, target], args.unit, args.feature,
                  stopwords=args.n_stopwords,
                  stopword_basis=args.stopword_basis,
                  score_basis=args.score_basis,
                  frequency_basis=args.frequency_basis,
                  max_distance=args.max_distance,
                  distance_metric=args.distance_metric,
                  min_score=args.min_score,
                  parallel=args.parallel)
    end = time.time() - start
    search.texts = texts
    search.parameters = params
    search.matches = matches
    print(f'Search found {len(matches)} matches in {end}s.')
    connection.insert(matches)
    connection.update(search)
    matches.sort(key=lambda x: x.score, reverse=True)
    print('The Top 10 Matches')
    print('------------------')
    print()
    print("Result\tScore\tSource Locus\tTarget Locus\tShared")
    for i, m in enumerate(matches[:10]):
        units = connection.find('units', _id=[m.source_unit, m.target_unit])
        shared = m.matched_features
        print(f'{i}.\t{m.score}\t{units[0].tags[0]}\t{units[1].tags[0]}\t{[t for t in shared]}')


if __name__ == '__main__':
    main()
