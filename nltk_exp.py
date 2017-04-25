import nltk
from collections import Counter

sentence = """This is a sentence with a duplicate word."""
tokens = nltk.word_tokenize(sentence)
pos = nltk.pos_tag(tokens)
# nltk.help.upenn_tagset('VBG') # Help on POS

# dpos = dict(pos)
# revd = {v: k for k, v in dpos.items()} # Reverses keys/values
# [item for item in pos if 'NN' in item] # Stuff