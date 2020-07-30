"""Tools for exporting searches as XML.

Functions
---------
build
  Construct an XML tree from a completed Tesserae search.
dump
  Dump a Tesserae search to file as XML.
dumps
  Dump a Tesserae search to an XML string.
format_result
  Convert a search result into an XML element.

Notes
-----
The XML tree defined here is based on the XML output of v3.
"""
import math
import xml.etree.ElementTree as ET

from tesserae.db.entities import Match, Search, Text, Unit
from tesserae.utils.exports.highlight import highlight_matches
from tesserae.utils.paging import Pager
from tesserae.utils.search import get_max_score

  
def build(connection, search, source, target):
  """Construct an XML tree from a completed Tesserae search.

  Parameters
  ----------
  connection : tesserae.db.TessMongoConnection
    Connection to the MongoDB instance.
  search_id : str or `bson.objectid.ObjectID`
    The database id of the search to serialize.
  search : `tesserae.db.entities.Search`
    Search metadata.
  source : `tesserae.db.entities.Text`
  target : `tesserae.db.entities.Text`
    Source and target text data.
  
  Returns
  -------
    root : `xml.etree.ElementTree.Element`
      Root node of the XML tree with search metadata and results inside.
  """
  max_score = get_max_score(connection, search.id)
  pages = Pager(connection, search.id)

  # The root element contains most search parameters as attributes.
  root = ET.Element(
    'results',
    attrib={
      'source': f'{source.author.lower()}.{source.title.lower().replace(" ", "_")}',
      'target': f'{target.author.lower()}.{target.title.lower().replace(" ", "_")}',
      'unit': search.parameters['source']['unit'].lower(),
      'feature': search.parameters['method']['feature'].lower(),
      'sessionID': search.id,
      'stop': len(search.parameters['method']['stopwrods']),
      'stbasis': '',
      'max_dist': f'{search.parameters["method"]["max_distance"]}',
      'dibasis': f'{search.parameters["method"]["distance_basis"]}',
      'cutoff': f'{0}',
    }
  )

  # v3 included comments with the Tesserae version and stopwords.
  ET.SubElement(root, 'comment', text='V5 Results.')
  ET.SubElement(root, 'commmonwords',
                text=', '.join(search.stopwords))

  # Each result is converted to an XML element and added to the tree.
  for page in pages:
    for result in page:
      format_result(root, result, max_score)
  
  return root


def dump(filename, connection, search, source, target):
  """Dump a Tesserae search to file as XML.

  Parameters
  ----------
  filepath : str
    Path to the output XML file.
  connection : tesserae.db.TessMongoConnection
    Connection to the MongoDB instance.
  search : `tesserae.db.entities.Search`
    Search metadata.
  source : `tesserae.db.entities.Text`
  target : `tesserae.db.entities.Text`
    Source and target text data.
  """
  # The xml module does not have a native file write, so first convert to
  # string, then dump the string to file.
  out = dumps(connection, search, source, target)
  with open(filename, 'w') as f:
    f.write(out)


def dumps(connection, search, source, target):
  """Dump a Tesserae search to an XML string.

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
    out : str
      XML string with search metadata and results.
  """
  out = build(connection, search)
  return ET.tostring(out)


def format_result(root, match, max_score):
  """Convert a search result into an XML element.

  Parameters
  ----------
  root : xml.etree.ElementTree.Element
    The root element of the XML tree being constructed.
  match : tesserae.db.entities.Match
    The result to serialize.
  max_score : float
    The max observed score in the search. Required for normalization.
  """
  # This element contains the match word and score data as attributes and the
  # source and target units as sub-elements.
  tessdata = ET.SubElement(root, 'tessdata', attrib={
    'keywords': ', '.join(match.matched_features),
    'score': f'{match.score * 10 / max_score}',
    'raw_score': f'{match.score}'
  })

  # Match words are highlighted with span tags in v3 XML.
  markup = ('<span class="matched">', '</span>')

  # Add the source unit element to the result with locus data as attributes
  # and the text with highlights as inner text.
  ET.SubElement(tessdata, 'phrase',
    attrs={
      'text': 'source',
      'work': ' '.join(match.source_tag.split()[:-1]),
      'unitId': str(match.source_unit)
                if not isinstance(match.source_unit, Unit)
                else str(match.source_unit.id),
      'line': match.source_tag.split()[-1]
    },
    text=highlight_matches(
      match.source_snippet,
      [i[0] for i in match.highlight],
      markup=markup
    )
  )

  # Add the target unit element to the result with locus data as attributes
  # and the text with highlights as inner text.
  ET.SubElement(tessdata, 'phrase',
    attrs={
      'text': 'target',
      'work': ' '.join(match.target_tag.split()[:-1]),
      'unitId': str(match.target_unit)
                if not isinstance(match.target_unit, Unit)
                else str(match.target_unit.id),
      'line': match.target_tag.split()[-1]
    },
    text=highlight_matches(
      match.target_snippet,
      [i[0] for i in match.highlight],
      markup=markup)
  )
