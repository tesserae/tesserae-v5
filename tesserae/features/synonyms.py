from tesserae.data import load_synonym_dictionary
from tesserae.features.lemmata import get_lemmatizer


def get_synonymifier(language, feature):
    """Retrieve correct synonym extraction function

    Returns
    -------
    List[str] -> List[List[str]]
        a function that takes a list of tokens and returns extracted synonym
        features for each token
    """
    lemmatizer = get_lemmatizer(language)
    syn_dict = load_synonym_dictionary(language, feature)

    def synonymify(tokens):
        result = []
        lemmata = lemmatizer.lookup(tokens)
        for lemma in lemmata:
            if not lemma[1]:
                result.append([lemma[0]])
                continue
            cur_set = set()
            for lem in lemma[1]:
                lem_lemma = lem[0]
                if lem_lemma in syn_dict:
                    cur_set.update(syn_dict[lem_lemma])
                else:
                    cur_set.add(lem_lemma)
            result.append(list(cur_set))
        return result

    return synonymify
