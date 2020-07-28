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

from tesserae.db.entities import Match, Search, Text, Unit
from tesserae.utils.exports.highlight import highlight_matches


def build(connection, search_id):
  """Construct a JSON object from a completed Tesserae search.

  Parameters
  ----------
  connection : tesserae.db.TessMongoConnection
    Connection to the MongoDB instance.
  search_id : str or `bson.objectid.ObjectID`
    The database id of the search to serialize.
  
  Returns
  -------
    obj : dict
      JSON-compatible dictionary with search metadata and results.
  """
  search = connection.find(Search.collection, id=search_id)
  results = connection.find(Match.collection, search_id=search_id)
  source = connection.find(Text.collection,
                           id=search.parameters['source']['object_id'])
  target = connection.find(Text.collection,
                           id=search.parameters['target']['object_id'])
  
  results.sort(key=lambda x: x.score, reverse=True)
  max_score = max(results[0].score, 10)
  min_score = int(math.floor(results[-1].score))

  out = search.json_encode()
  out['parameters']['source'].update(source.json_encode())
  out['parameters']['target'].update(target.json_encode())
  out['results'] = list(
    itertools.starmap(
      format_result, 
      zip(results, itertools.repeat(max_score, len(results)))
    )
  )
  del out['results_id']
  del out['progress']
  del out['status']
  del out['msg']
  
  return out


def dump(connection, search_id, filepath):
  """Dump a Tesserae search to file as JSON.

  Parameters
  ----------
  connection : tesserae.db.TessMongoConnection
    Connection to the MongoDB instance.
  search_id : str or `bson.objectid.ObjectID`
    The database id of the search to serialize.
  filepath : str
    Path to the output JSON file.
  """
  out = build(connection, search_id)
  with open(filepath, 'w') as f:
    json.dump(out, f)


def dumps(connection, search_id):
  """Dump a Tesserae search to a JSON string.

  Parameters
  ----------
  connection : tesserae.db.TessMongoConnection
    Connection to the MongoDB instance.
  search_id : str or `bson.objectid.ObjectID`
    The database id of the search to serialize.
  
  Returns
  -------
    obj : str
      JSON string with search metadata and results.
  """
  out = build(connection, search_id)
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