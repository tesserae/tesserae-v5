import re

def tri_greek(tokens):
	pattern = re.compile(r'\w')
	token_grams = []
	token_tri = {}
	for token in tokens:
		grams = []
		characters = []
		for a in token:
			char = pattern.match(a)
			if char != None:
				char = char.group()
				characters.append(char)
		final = len(characters) - 1
		if len(characters) < 3:
			grams = [' ']
			#empty string conflicts with create_features in tesserae/tokenizers/base.py
		else:
			for a in range(final-1):
				grams.append(characters[a]+characters[a+1]+characters[a+2])
		token_grams.append(grams)
		token_tri[token] = grams
#	print(token_tri)
	return token_grams

def tri_latin(tokens):
	pattern = re.compile(r'\w')
	token_grams = []
	for token in tokens:
		grams = []
		characters = []
		for a in token:
			char = pattern.match(a)
			if char != None:
				char = char.group()
				characters.append(char)
		final = len(characters) - 1
		if len(characters) < 3:
			grams = [' ']
			#empty string conflicts with create_features in tesserae/tokenizers/base.py
		else:
			for a in range(final-1):
				grams.append(characters[a]+characters[a+1]+characters[a+2])
		token_grams.append(grams)
	return token_grams

