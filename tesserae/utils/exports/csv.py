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

from tesserae.db.entities import Match, Search, Text, Unit
from tesserae.utils.exports.highlight import highlight_matches


def build(connection, search_id, stream, delimiter=','):
  """Construct CSV from a completed Tesserae search.

  Parameters
  ----------
  connection : tesserae.db.TessMongoConnection
    Connection to the MongoDB instance.
  search_id : str or `bson.objectid.ObjectID`
    The database id of the search to serialize.
  stream : io.TextIOBase
    Text stream to write to, usually a string or file stream
  delimiter : str, optional
    The row iterm separator. Default: ','.
  
  Returns
  -------
    stream : io.TextIOBase
      The same object passed to ``stream``.
  """
  # Pull the search and text data from the database.
  search = connection.find(Search.collection, id=search_id)
  results = connection.find(Match.collection, search_id=search_id)
  source = connection.find(Text.collection,
                           id=search.parameters['source']['object_id'])
  target = connection.find(Text.collection,
                           id=search.parameters['target']['object_id'])
  
  # Sort the search results by score and get the max and min scores.
  results.sort(key=lambda x: x.score, reverse=True)
  max_score = max(results[0].score, 10)
  min_score = int(math.floor(results[-1].score))

  # The search parameters and metadata are written as comments to the top of
  # the CSV stream.
  comments = [
    f'# Tesserae V5 Results',
    f'#',
    f'# session\t= {search_id}',
    f'# source\t= {source.author}.{source.title.lower().replace(" ", "_")}',
    f'# target\t= {target.author}.{target.title.lower().replace(" ", "_")}',
    f'# unit\t= {search.parameters["source"]["unit"]}',
    f'# feature\t= {search.parameters["method"]["feature"]}',
    f'# stopsize\t= {len(search.parameters["method"]["stopwords"])}',
    f'# stbasis\t= ',
    f'# stopwords\t= {search.parameters["method"]["stopwords"]}',
    f'# max_dist\t= {search.parameters["method"]["max_distance"]}',
    f'# dibasis\t= {search.parameters["method"]["distance_basis"]}',
    f'# cutoff\t= {min_score}',
    f'# filter\t= off',
  ]

  stream.write('\n'.join(comments))

  # Add CSV rows to the stream from dictionary versions of the results.
  # Row headers are passed in the second argument.
  writer = csv.DictWriter(
    stream,
    ["Result", "Target_Loc", "Target_Txt",
     "Source_Loc", "Source_Txt", "Shared",
     "Score", "Raw_Score"],
    delimiter=delimiter
  )
  writer.writeheader()

  # Convert each result to a dict with keys corresponding to the headers
  # passed to ``writer``.
  results = zip(results,
                range(1, len(results) + 1),
                itertools.repeat(max_score, len(results)))
  writer.writerows(itertools.starmap(format_result, results))
  
  return stream


def dump(connection, search_id, filepath, delimiter=','):
  """Dump a Tesserae search to file as CSV.

  Parameters
  ----------
  connection : tesserae.db.TessMongoConnection
    Connection to the MongoDB instance.
  search_id : str or `bson.objectid.ObjectID`
    The database id of the search to serialize.
  filepath : str
    Path to the output CSV file.
  delimiter : str, optional
    The row iterm separator. Default: ','.
  """
  with open(filepath, 'w') as f:
    build(connection, search_id, f, delimiter=delimiter)


def dumps(connections, search_id, delimiter):
  """Dump a Tesserae search to an CSV string.

  Parameters
  ----------
  connection : tesserae.db.TessMongoConnection
    Connection to the MongoDB instance.
  search_id : str or `bson.objectid.ObjectID`
    The database id of the search to serialize.
  delimiter : str, optional
    The row iterm separator. Default: ','.
  
  Returns
  -------
    out : str
      CSV string with search metadata and results.
  """
  output = io.StringIO()
  build(connections, search_id, output, delimiter=delimiter)
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