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
from tesserae.matchers import SparseMatrixSearch
from tesserae.matchers.text_options import TextOptions
from tesserae.utils.search import \
    check_cache, NORMAL_SEARCH, get_results, PageOptions, _run_search
from tesserae.utils.stopwords import create_stoplist, get_stoplist_tokens


def parse_args(args=None):
    p = argparse.ArgumentParser(prog='tesserae.cli.search',
                                description='Perform a Tesserae search')

    db = p.add_argument_group(title='database',
                              description='database connection details')
    search = p.add_argument_group(
        title='search',
        description=(
            'search parameters; use underscores (_) in place of spaces '
            'if necessary'))

    db.add_argument('--user',
                    type=str,
                    default=None,
                    help='user to access the database as')
    db.add_argument('--password',
                    action='store_true',
                    help='ask to be prompted for a database password')
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

    search.add_argument('source_author',
                        type=str,
                        help='author of the source text')
    search.add_argument('source_title',
                        type=str,
                        help='title of the source text')
    search.add_argument('source_unit',
                        type=str,
                        choices=['line', 'phrase'],
                        help='units to match on the source text')
    search.add_argument('target_author',
                        type=str,
                        help='author of the target text')
    search.add_argument('target_title',
                        type=str,
                        help='title of the target text')
    search.add_argument('target_unit',
                        type=str,
                        choices=['line', 'phrase'],
                        help='units to match on the target text')
    search.add_argument('--feature',
                        type=str,
                        choices=['form', 'lemmata'],
                        default='lemmata',
                        help='feature to match on')
    search.add_argument('--n-stopwords',
                        type=int,
                        default=10,
                        help='size of the stopwords list')
    search.add_argument('--stopword-basis',
                        choices=['corpus', 'texts'],
                        default='corpus',
                        help='data to use in computing a stopwords list')
    search.add_argument('--score-basis',
                        choices=['word', 'stem'],
                        default='word',
                        help='atom to consider in computing scores')
    search.add_argument('--freq-basis',
                        choices=['corpus', 'texts'],
                        default='texts',
                        help='data to use in computing frequencies')
    search.add_argument(
        '--max-distance',
        type=int,
        default=999,
        help='maximum allowable distance between matched tokens')
    search.add_argument('--distance-basis',
                        choices=['span', 'frequency'],
                        default='frequency',
                        help='how to compute distance between matched tokens')
    search.add_argument('--min-score',
                        type=int,
                        default=0,
                        help='lowest scoring match to keep')

    search.add_argument('--output', type=str, default=None,
                        help='path to write results to file')
    search.add_argument('--output-format', choices=['csv', 'json', 'tab', 'xml'],
                        help='format to write the results in')
    
    return p.parse_args(args)


def main():
    """Perform Tesserae search and display the top 10 results"""
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

    source_author = args.source_author.lower().replace('-', ' ')
    source_title = args.source_title.lower().replace('-', ' ')
    source = TextOptions(text=connection.find('texts',
                                              author=source_author,
                                              title=source_title)[0],
                         unit_type=args.source_unit)
    target_author = args.target_author.lower().replace('_', ' ')
    target_title = args.target_title.lower().replace('_', ' ')
    target = TextOptions(text=connection.find('texts',
                                              author=target_author,
                                              title=target_title)[0],
                         unit_type=args.target_unit)

    start = time.time()
    stopword_indices = create_stoplist(
        connection,
        args.n_stopwords,
        args.feature,
        source.text.language,
        basis='corpus' if args.stopword_basis == 'corpus' else
        [source.text.id, target.text.id])
    stopword_tokens = get_stoplist_tokens(connection, stopword_indices,
                                          args.feature, source.text.language)
    parameters = {
        'source': {
            'object_id': str(source.text.id),
            'units': source.unit_type
        },
        'target': {
            'object_id': str(target.text.id),
            'units': target.unit_type
        },
        'method': {
            'name': SparseMatrixSearch.matcher_type,
            'feature': args.feature,
            'stopwords': stopword_tokens,
            'freq_basis': args.freq_basis,
            'max_distance': args.max_distance,
            'distance_basis': args.distance_basis
        }
    }
    results_id = check_cache(connection, parameters['source'],
                             parameters['target'], parameters['method'])
    if results_id:
        print('Cached results found.')
        search = connection.find(Search.collection,
                                 results_id=results_id,
                                 search_type=NORMAL_SEARCH)[0]
    else:
        search = Search(results_id=uuid.uuid4().hex,
                        search_type=NORMAL_SEARCH,
                        parameters=parameters)
        connection.insert(search)
        search_params = {
            'source': source,
            'target': target,
            'feature': parameters['method']['feature'],
            'stopwords': parameters['method']['stopwords'],
            'freq_basis': parameters['method']['freq_basis'],
            'max_distance': parameters['method']['max_distance'],
            'distance_basis': parameters['method']['distance_basis'],
            'min_score': 0
        }
        _run_search(connection, search, SparseMatrixSearch.matcher_type,
                    search_params)
    matches = get_results(
        connection,
        search.id,
        PageOptions(
            sort_by='score',
            sort_order='descending',
            per_page=10,
            page_number=0
    ))
    end = time.time() - start
    matches.sort(key=lambda x: x['score'], reverse=True)
    print(f'Search found {len(matches)} matches in {end}s.')
    display_count = 10 if len(matches) >= 10 else len(matches)
    print(f'The Top {display_count} Matches')
    print('------------------')
    print()
    print("Result\tScore\tSource Locus\tTarget Locus\tShared")
    for i, m in enumerate(matches[:10]):
        shared = m['matched_features']
        print(f'{i}.\t{m["score"]}\t{m["source_tag"]}\t{m["target_tag"]}\t'
              f'{[t for t in shared]}')


if __name__ == '__main__':
    main()
