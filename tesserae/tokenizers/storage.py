from tesserae.db import convert_to_entity, create_filter, Token


class InvalidTokenError(Exception):
    def __init__(self, token):
        msg = 'The supplied token {} is not a valid token type or does not'
        msg += ' contain the correct fields.'
        super(InvalidTokenError, self).__init__(msg.format(token))


class DuplicateTokenError(Exception):
    """Raised when duplicate tokens exist in the database"""
    def __init__(self, token):
        msg = 'Multiple tokens {} exist in the database.'.format(token)
        super(DuplicateTokenError, self).__init__(msg)


class NoTokenError(Exception):
    """Raised when attempting to update a token that is not in the database"""
    def __init__(self, token):
        msg = 'Token {} does not exist in the database.'.format(token)
        super(NoTextError, self).__init__(msg)


@convert_to_entity(Token)
def retrieve_token_list(client, raw=None, type=None, lemmata=None):
    """Get a list of tokens from the database.

    Parameters
    ----------
    client : `pymongo.MongoClient`
        Client connection to the database.
    raw : str or list of str, optional
        Filter the list by the supplied raw word string(s).
    type : str or list of str, optional
        Filter the list by the supplied normalized word string(s).
    lemmata : str or list of str, optional
        Filter the list by the supplied lemma(s).

    Returns
    -------
    tokens : list of `tesserae.db.Token`
        The list of tokens that match the filter conditions.

    """
    # Create the filter
    filter = create_filter(raw=raw, type=type)
    if lemmata is not None:
        lemma_filter = {
            'lemmata': {
                '$exists': True, '$elemMatch': {'$in': lemmata}}}

        if '$and' in filter:
            filter['$and'].append(lemma_filter)
        else:
            filter.update(lemma_filter)

    # Retrieve the texts and put them into a nice format
    docs = client['tokens'].find(filter)
    tokens = [doc for doc in docs]
    return tokens


def insert_tokens(client, tokens):
    """Insert one or more tokens into the database.

    Parameters
    ----------
    client : `pymongo.MongoClient`
        Client connection to the database.
    tokens : list of dict or list of tesserae.db.Token
        The tokens to insert. If dicts are supplied, they will be converted to
        Token for validation.

    Returns
    -------
    result : `pymongo.results.InsertManyResult`

    Raises
    ------
    InvalidTokenError
        Raised when a token with invalid type or fields is encountered.
    """
    # Standardize all tokens as Token entities
    entities = []
    raws = []
    for t in tokens:
        if not isinstance(t, Token):
            try:
                entities.append(Token(**t))
                raws.append(t['raw'])
            except (TypeError, KeyError, AttributeError):
                raise InvalidTokenError(t)
        else:
            entities.append(t)
            raws.append(t.raw)

    # Only insert tokens that do not already exist in the database
    db_tokens = retrieve_token_list(client, raw=raws)
    del raws

    for t, i in enumerate(entities):
        if t.raw == db_tokens[i].raw:
            entities.remove(t)

    result = client.tokens.insert_many([e.json_encode() for e in entities])
    return result


def update_token(client, token):
    """Update a token's database entry.

    Parameters
    ----------
    client : `pymongo.MongoClient`
        Client connection to the database.
    token : dict or tesserae.db.Token
        The token to update. If a dict is supplied, it will be converted to
        Token for validation.

    Returns
    -------
    result : `pymongo.results.UpdateOneResult`

    See Also
    --------
    tesserae.db.Token

    Raises
    ------
    InvalidTokenError
        Raised when a token with invalid type or fields is encountered.
    NoTokenError
        Raised when trying to update a token that does not exist.
    DuplicateTokenError
        Raised when duplicate tokens are encountered during the update.
    """
    # Standardize the dictionary as a Token
    if not isinstance(token, Token):
        try:
            token = Token(**token)
        except (TypeError, KeyError, AttributeError):
            raise InvalidTokenError(token)

    # Ensure that the token to update already exists and that no duplicates
    # exist.
    db_token = retrieve_token_list(client, raw=token.raw)

    if len(db_token) == 0:
        raise NoTokenError(token)
    elif len(db_token) > 1:
        raise DuplicateTokenError(token)

    # Perform the update.
    result = client.tokens.update_one(
        {'raw': token.raw},
        {'$set': token.json_encode(exclude=['_id', 'raw'])})

    return result


def update_token_frequencies(client, token_):
    pass
