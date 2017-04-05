import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

data = pd.read_csv("train.csv").fillna("")
sentence = data["question1"][0]
tokens = nltk.word_tokenize(sentence)


# Check for word similarity as a feature, then display a graph how it matches.
# Why is nltk not working?