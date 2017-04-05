import pandas as pd
import matplotlib.pyplot as plt
import nltk # Install nltk and install the 'punkt' from models, using nltk.download()

data = pd.read_csv("train.csv").fillna("")
n = data.shape[0]
#n = 60000 # Reduced it because tokenizations take forever

# Check for word similarity as a feature, then display a graph how it matches.

q1_tok, q2_tok, num_common_words, rel_sum = [], [], [], []
for i in range(n):
    q1_tok.append(nltk.word_tokenize(data["question1"][i]))
    q2_tok.append(nltk.word_tokenize(data["question2"][i]))
    num_common_words.append(len(set(q1_tok[i]) & set(q2_tok[i])))
    rel_sum.append(num_common_words[i] / (len(q1_tok[i]) + len(q2_tok[i])))

comparison = {'is_duplicate': data["is_duplicate"][0:n] ,
              'num_common': num_common_words,
              'rel_sum': rel_sum
              }
# Number of common words / (total number of words in the two questions) vs. Number of common words
comparison = pd.DataFrame(comparison)
# We have to figure out how to save these preprocessed data strings

plt.figure()
plt.scatter(comparison['rel_sum'], comparison['num_common'], c = comparison['is_duplicate'], s=[[20 for x in range(n)]])
plt.show()

# This seems to be a good feature. Some ideas:
# Find number of unique verbs & nouns as a feature
# How many close (meaning-wise!!) nouns & verbs are there to the unique ones.
# Identify specific words that appear often in duplicates.
# Look for other intuitive semantic stuff.