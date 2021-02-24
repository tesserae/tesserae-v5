"""Tools for highlighting match words in units.

Functions
---------
highlight_matches
  Highlight match tokens in a search result.
"""
import re


def highlight_matches(snippet, match_indices, markup='**'):
    """Highlight match tokens in a search result.

    Tesserae file dumps preserve match token highlighting in the raw unit text.
    Given a snippet and match token indices, wraps the tokens at the indices
    with markup to highlight it. Token characters include only alphabetic and
    diacritical ASCII and UTF-8 characters.

    Parameters
    ----------
    snippet : str
        The raw text of the match unit to highlight.
    match_indices : sequence of int
        The indices of tokens to highlight.
    markup : str or tuple of str, optional
        Markup to apply to wrap match tokens in. If a string, the string is
        placed on both sides of the token. If a tuple, the 0th element is
        placed at the start and 1st element is placed at the end.

    Returns
    -------
    highlighted : str
        A copy of ``snippet`` with the metch tokens highlighted by ``markup``.

    Examples
    --------
    >>> highlight_matches('foo !bar, baz-quux.', [1, 3])
    'foo !**bar**, baz-**quux**.'
    >>> highlight_matches('foo !bar, baz-quux', [1, 3], markup='@')
    'foo !@bar@, baz-@quux@'
    >>> highlight_matches('foo !bar, baz-quux', [1, 3], markup=('<b>', '</b>'))
    'foo !<b>bar</b>, baz-<b>quux</b>'
    """
    if isinstance(markup, str):
        markup = (markup, markup)

    if not match_indices or markup[0] == '':
        return snippet
    elif isinstance(match_indices, int):
        match_indices = [match_indices]

    match_indices.sort()
    highlighted = []

    # Split between word- and non-word tokens but only count the word tokens
    # when evaluating the match indices.
    tokens = re.split(r'([\s\d.!,;?\-&]+)', snippet, flags=re.UNICODE)
    tidx = 0
    midx = match_indices.pop(0)

    # If looking at a word token, highlight it if it is in the position of the
    # next match token and pull the next match index. Always increment the word
    # token counter. Otherwise, append the token unaltered.
    for i, t in enumerate(tokens):
        if re.search(r'[\w]+', t):
            highlighted.append(f'{markup[0]}{t}{markup[1]}' if midx ==
                               tidx else t)
            midx = match_indices.pop(
                0) if midx == tidx and match_indices else midx
            tidx += 1
        else:
            highlighted.append(t)

    # Spaces and punctuation should be preserved in non-word tokens, so join on
    # the empty string to recreate the snippet on with highlights.
    return ''.join(highlighted)
