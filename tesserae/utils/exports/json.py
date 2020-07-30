"""Tools for exporting searches as JSON objects.

Functions
---------
build
  Construct a JSON object from a completed Tesserae search.
dump
  Dump a Tesserae search to file as JSON.
dumps
  Dump a Tesserae search to a JSON string.
format_result
  Convert a search result into a JSON object.
"""
import itertools
import json
import math

from tesserae.utils.exports.highlight import highlight_matches
from tesserae.utils.paging import Pager
from tesserae.utils.search import get_max_score


def build(connection, search, source, target):
  """Construct a JSON object from a completed Tesserae search.

  Parameters
  ----------
  connection : tesserae.db.TessMongoConnection
    Connection to the MongoDB instance.
  search : `tesserae.db.entities.Search`
    Search metadata.
  source : `tesserae.db.entities.Text`
  target : `tesserae.db.entities.Text`
    Source and target text data.
  
  Returns
  -------
    obj : dict
      JSON-compatible dictionary with search metadata and results.
  """
  max_score = get_max_score(connection, search.id)
  pages = Pager(connection, search.id)

  out = search.json_encode()
  out['parameters']['source'].update(source.json_encode())
  out['parameters']['target'].update(target.json_encode())
  out['results'] = []
  for page in pages:
    out['results'].extend(
      itertools.starmap(
        format_result, 
        zip(page, itertools.repeat(max_score, len(page)))
      )
    )
  del out['results_id']
  del out['progress']
  del out['status']
  del out['msg']
  
  return out


def dump(filepath, connection, search, source, target):
  """Dump a Tesserae search to file as JSON.

  Parameters
  ----------
  filepath : str
    Path to the output JSON file.
  connection : tesserae.db.TessMongoConnection
    Connection to the MongoDB instance.
  search : `tesserae.db.entities.Search`
    Search metadata.
  source : `tesserae.db.entities.Text`
  target : `tesserae.db.entities.Text`
    Source and target text data.
  """
  out = build(connection, search, source, target)
  with open(filepath, 'w') as f:
    json.dump(out, f)


def dumps(connection, search, source, target):
  """Dump a Tesserae search to a JSON string.

  Parameters
  ----------
  connection : tesserae.db.TessMongoConnection
    Connection to the MongoDB instance.
  search : `tesserae.db.entities.Search`
    Search metadata.
  source : `tesserae.db.entities.Text`
  target : `tesserae.db.entities.Text`
    Source and target text data.
  
  Returns
  -------
    obj : str
      JSON string with search metadata and results.
  """
  out = build(connection, search, source, target)
  return json.dumps(out)


def format_result(match, max_score):
  """Convert a search result into a JSON object.

  Parameters
  ----------
  match : tesserae.db.entities.Match
    The result to serialize.
  max_score : float
    The max observed score in the search. Required for normalization.

  Returns
  -------
  obj : dict
    The result as a JSON-compatible dictionary.
  """
  out = match.json_encode()
  out['result_id'] = out['_id']
  del out['_id']
  out['source_snippet'] = highlight_matches(out['source_snippet'], [i[0] for i in match.match_indices])
  out['target_snippet'] = highlight_matches(out['target_snippet'], [i[1] for i in match.match_indices])
  del out['match_indices']
  out['score'] = match.score * 10 / max_score
  out['raw_score'] = match.score
  return out