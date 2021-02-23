"""Tools for exporting searches in CSV format.

Functions
---------
build
  Construct CSV from a completed Tesserae search.
dump
  Dump a Tesserae search to file as CSV.
dumps
  Dump a Tesserae search to a CSV string.
format_result
  Convert a search result into a CSV row.
"""
import csv
import io
import itertools
import math

from tesserae.utils.exports.highlight import highlight_matches
from tesserae.utils.paging import Pager
from tesserae.utils.search import get_max_score


def build(stream, connection, search, source, target, delimiter=','):
    """Construct CSV from a completed Tesserae search.

    Parameters
    ----------
    stream : io.TextIOBase
        Text stream to write to, usually a string or file stream
    connection : tesserae.db.TessMongoConnection
        Connection to the MongoDB instance.
    search : `tesserae.db.entities.Search`
        Search metadata.
    source : `tesserae.db.entities.Text`
    target : `tesserae.db.entities.Text`
        Source and target text data.
    delimiter : str, optional
        The row iterm separator. Default: ','.

    Returns
    -------
        stream : io.TextIOBase
            The same object passed to ``stream``.
    """
    max_score = get_max_score(connection, search.id)
    pages = Pager(connection, search.id)

    source_title = source.title.lower().replace(" ", "_")
    target_title = target.title.lower().replace(" ", "_")
    # The search parameters and metadata are written as comments to the top of
    # the CSV stream.
    comments = [
        '# Tesserae V5 Results',
        '#',
        f'# session   = {search.id}',
        f'# source    = {source.author}.{source_title}',
        f'# target    = {target.author}.{target_title}',
        f'# unit      = {search.parameters["source"]["units"]}',
        f'# feature   = {search.parameters["method"]["feature"]}',
        f'# stopsize  = {len(search.parameters["method"]["stopwords"])}',
        f'# stbasis   = ',
        f'# stopwords = {search.parameters["method"]["stopwords"]}',
        f'# max_dist  = {search.parameters["method"]["max_distance"]}',
        f'# dibasis   = {search.parameters["method"]["distance_basis"]}',
        f'# cutoff    = {0}',
        f'# filter    = off',
    ]

    stream.write('\n'.join(comments))
    stream.write('\n')

    # Add CSV rows to the stream from dictionary versions of the results.
    # Row headers are passed in the second argument.
    writer = csv.DictWriter(stream, [
        "Result", "Target_Loc", "Target_Txt", "Source_Loc", "Source_Txt",
        "Shared", "Score", "Raw_Score"
    ],
                            delimiter=delimiter)
    writer.writeheader()

    for i, page in enumerate(pages):
        # Convert each result to a dict with keys corresponding to the headers
        # passed to ``writer``.
        start = i * pages.per_page + 1
        end = start + len(page)
        results = zip(page, range(start, end), itertools.repeat(max_score))
        writer.writerows(itertools.starmap(format_result, results))

    return stream


def dump(filename, connection, search, source, target, delimiter):
    """Dump a Tesserae search to file as CSV.

    Parameters
    ----------
    filename : str
        Path to the output CSV file.
    connection : tesserae.db.TessMongoConnection
        Connection to the MongoDB instance.
    search : `tesserae.db.entities.Search`
        Search metadata.
    source : `tesserae.db.entities.Text`
    target : `tesserae.db.entities.Text`
        Source and target text data.
    delimiter : str, optional
        The row iterm separator. Default: ','.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        build(f, connection, search, source, target, delimiter=delimiter)


def dumps(connection, search, source, target, delimiter):
    """Dump a Tesserae search to an CSV string.

    Parameters
    ----------
    connection : tesserae.db.TessMongoConnection
        Connection to the MongoDB instance.
    search : `tesserae.db.entities.Search`
        Search metadata.
    source : `tesserae.db.entities.Text`
    target : `tesserae.db.entities.Text`
        Source and target text data.
    delimiter : str, optional
        The row iterm separator. Default: ','.

    Returns
    -------
        out : str
            CSV string with search metadata and results.
    """
    output = io.StringIO(newline='')
    build(output, connection, search, source, target, delimiter=delimiter)
    return output.getvalue()


def format_result(match, idx, max_score):
    """Convert a search result into a CSV row.

    Parameters
    ----------
    match : MatchResult
        The result to serialize.
    idx : int
        The row number of this result.
    max_score : float
        The max observed score in the search. Required for normalization.

    Returns
    -------
    obj : dict
        The result converted into a CSV DictWriter compatible format.
    """
    source_txt = highlight_matches(match['source_snippet'],
                                   [i[0] for i in match['highlight']])
    target_txt = highlight_matches(match['target_snippet'],
                                   [i[1] for i in match['highlight']])
    features = '; '.join(match['matched_features'])
    target_loc = match['target_tag']
    source_loc = match['source_tag']
    score = match['score']
    return {
        'Result': f'{idx}',
        'Target_Loc': f'\"{target_loc}\"',
        'Target_Txt': f'\"{target_txt}\"',
        'Source_Loc': f'\"{source_loc}\"',
        'Source_Txt': f'\"{source_txt}\"',
        'Shared': f'\"{features}\"',
        'Score': f'{score * 10 / max_score}',
        'Raw_Score': f'{score}'
    }
