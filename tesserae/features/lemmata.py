from cltk.semantics.latin.lookup import Lemmata

_LEM_MAPPER = {
    'latin': Lemmata('lemmata', 'lat'),
    'greek': Lemmata('lemmata', 'grc')
}


def get_lemmatizer(language):
    return _LEM_MAPPER[language]
