from tesserae.utils import TessFile
from tesserae.db import TessMongoConnection, Entity, Text
from tesserae.tokenizers import GreekTokenizer, LatinTokenizer
from tesserae.unitizer import Unitizer


conn = TessMongoConnection('127.0.0.1', 27017, None, None, 'tesstest')
# tess = TessFile('demo/la/vergil.aeneid.tess')
tess = TessFile('demo/la/cicero.de_oratore.tess')
tok = LatinTokenizer(conn)
# tess = TessFile('demo/grc/euripides.heracles.tess')
# tess = TessFile('demo/grc/plato.epistles.tess')
# tok = GreekTokenizer(conn)
unitizer = Unitizer()
# tokens, tags, fs, freq = tok.tokenize(tess.read(), text=Text(language='latin'))
tokens, tags, features = tok.tokenize2(tess.read(), text=Text(language='latin'))
lines, phrases = unitizer.unitize(tokens, tags, metadata=Text(language='latin'))

print([t.features for t in tokens[:30]])
print(''.join([t.display for t in tokens[:30]]))
print(' '.join([t.features['form'].token if 'form' in t.features else '' for t in tokens[:30]]))

#for i in range(10):
#    if tokens[0][i].feature_set is not None:
#        tag = tokens[0][i].feature_set
#        if not isinstance(tag, str):
#            tag = tag.form
#        print(tokens[0][i].display, tag)
#    else:
#        print(tokens[0][i].display)

#for t in tokens[0]:
#    if isinstance(t.feature_set, str):
#        print(t.display, t.feature_set)

print('Lines:   {}'.format(any([len(u.tags) == 0 for u in lines])))
print('Phrases: {}'.format(any([len(u.tags) == 0 for u in phrases])))

print([l.tags for l in lines[:100]])
print([p.tags for p in phrases[:100]])


print(''.join([str(phrases[-5].tags)] + [t.display for t in phrases[-5].tokens]))
print(''.join([str(phrases[-4].tags)] + [t.display for t in phrases[-4].tokens]))
print(''.join([str(phrases[-3].tags)] + [t.display for t in phrases[-3].tokens]))
print(''.join([str(phrases[-2].tags)] + [t.display for t in phrases[-2].tokens]))
print(''.join([str(phrases[-1].tags)] + [t.display for t in phrases[-1].tokens]))

# print(' '.join([t.feature_set.form for t in lines[0].tokens]))

#for line in lines:
#    if len(line.tags) == 0:
#        print(line.index)

#print('Phrases')
#for phrase in phrases:
#    if len(phrase.tags) == 0:
#        print(phrase.index)
