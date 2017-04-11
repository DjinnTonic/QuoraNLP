import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk # Install nltk and install the 'punkt' from models, using nltk.download()
from collections import Counter

data = pd.read_csv("train.csv").fillna("")
n = data.shape[0]

n = 6000 # Reduced it because tokenizations take forever

# Check for word similarity as a feature, then display a graph how it matches.

q1_tok, q2_tok, num_common_words, rel_sum, nc_nouns, nc_verbs = [], [], [], [], [], []

# Create a function for this instead of this cranky for-loop
for i in range(n):
    q1_tok.append(nltk.word_tokenize(data["question1"][i]))
    q2_tok.append(nltk.word_tokenize(data["question2"][i]))

    q1p, q2p = nltk.pos_tag(q1_tok[i]), nltk.pos_tag(q2_tok[i])
    com_dic = Counter(e[1] for e in set(q1p) & set(q2p)) # Decide which of these things you want returned as features from the function you will create
    nc_nouns.append(sum([v for k, v in com_dic.items() if 'NN' in k])    / (len(q1_tok[i]) + len(q2_tok[i]))   )  # Counts nouns, my genius code haha
    nc_verbs.append(sum([v for k, v in com_dic.items() if 'VB' in k])    / (len(q1_tok[i]) + len(q2_tok[i]))   )

    num_common_words.append(len(set(q1_tok[i]) & set(q2_tok[i])))
    rel_sum.append(num_common_words[i] / (len(q1_tok[i]) + len(q2_tok[i])))

comparison = {'is_duplicate': data["is_duplicate"][0:n] ,
              'num_common': num_common_words,
              'rel_sum': rel_sum + np.random.rand(n) / 20, # Randomization to see the plot points under each other
              'nc_nouns': nc_nouns + np.random.rand(n) / 20,
              'nc_verbs': nc_verbs + np.random.rand(n) / 20
              }
# Number of common words / (total number of words in the two questions) vs. Number of common words
comparison = pd.DataFrame(comparison)
# We have to figure out how to save these preprocessed data strings

plt.figure()
# plt.scatter(comparison['rel_sum'], comparison['num_common'], c = comparison['is_duplicate'], s=[[20 for x in range(n)]])
plt.scatter(comparison['nc_verbs'], comparison['nc_nouns'], c = comparison['is_duplicate'], s=[[20 for x in range(n)]])
plt.show()

# This seems to be a good feature. Some ideas:
# Find number of unique verbs & nouns as a feature
# How many close (meaning-wise!!) nouns & verbs are there to the unique ones.
# Identify specific words that appear often in duplicates.
# Look for other intuitive semantic stuff.
# Identify keywords.