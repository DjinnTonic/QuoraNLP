import nltk
sentence = """Soon we are going to a meeting about further developments."""
tokens = nltk.word_tokenize(sentence)
pos = nltk.pos_tag(tokens)
nltk.help.upenn_tagset('VBG')

dpos = dict(pos)
revd = {v: k for k, v in dpos.items()}
[item for item in pos if 'NN' in item]