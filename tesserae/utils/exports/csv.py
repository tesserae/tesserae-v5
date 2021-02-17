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

    # The search parameters and metadata are written as comments to the top of
    # the CSV stream.
    comments = [
        f'# Tesserae V5 Results',
        f'#',
        f'# session\t= {search.id}',
        f'# source\t= {source.author}.{source.title.lower().replace(" ", "_")}',
        f'# target\t= {target.author}.{target.title.lower().replace(" ", "_")}',
        f'# unit\t= {search.parameters["source"]["units"]}',
        f'# feature\t= {search.parameters["method"]["feature"]}',
        f'# stopsize\t= {len(search.parameters["method"]["stopwords"])}',
        f'# stbasis\t= ',
        f'# stopwords\t= {search.parameters["method"]["stopwords"]}',
        f'# max_dist\t= {search.parameters["method"]["max_distance"]}',
        f'# dibasis\t= {search.parameters["method"]["distance_basis"]}',
        f'# cutoff\t= {0}',
        f'# filter\t= off',
    ]

    stream.write('\n'.join(comments))

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
        start = i * 200 + 1
        end = start + 200
        results = zip(page, range(start, end),
                      itertools.repeat(max_score))
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
    with open(filename, 'w') as f:
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
    output = io.StringIO()
    build(output, connection, search, source, target, delimiter=delimiter)
    return output.getvalue()


def format_result(match, idx, max_score):
    """Convert a search result into a CSV row.

  Parameters
  ----------
  match : tesserae.db.entities.Match
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
    source_txt = highlight_matches(match.source_snippet,
                                   [i[0] for i in match.highlight])
    target_txt = highlight_matches(match.target_snippet,
                                   [i[1] for i in match.highlight])
    features = '; '.join(match.matched_features)
    return {
        'Result': f'{idx}',
        'Target_Loc': f'\"{match.target_tag}\"',
        'Target_Txt': f'\"{target_txt}\"',
        'Source_Loc': f'\"{match.source_tag}\"',
        'Source_Txt': f'\"{source_txt}\"',
        'Shared': f'\"{features}\"',
        'Score': f'{match.score * 10 / max_score}',
        'Raw_Score': f'{match.score}'
    }
