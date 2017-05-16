import pandas as pd
#pd.options.mode.chained_assignment = None
import numpy as np
import re
import nltk
from gensim.models import word2vec
from collections import Counter
import itertools
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#%matplotlib inline

#data = pd.read_csv("train.csv").sample(50000)

#STOP_WORDS = nltk.corpus.stopwords.words()

def clean_sentence(val):
    """remove chars that are not letters or numbers, downcase, then remove stop words
    Rami: Stop words might not be a good idea for the LSTM. Currently keep as-is.
    Later include somehting that realizes that for example I'm = I am ... etc.
    """
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")

    # for word in list(sentence):
    #     if word in STOP_WORDS:
    #         sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence


def clean_dataframe(data):
    "drop nans, then apply 'clean_sentence' function to question1 and 2"
    data = data.dropna(how="any")

    for col in ['question1', 'question2']:
        data[col] = data[col].apply(clean_sentence)

    return data


#data = clean_dataframe(data)
#h = 9
#data['question1'][data['is_duplicate']==1][0:h], data['question2'][data['is_duplicate']==1][0:h]

def build_corpus(data, col):
    """"Creates a list of lists containing words from each sentence
    Replaces rare words with <unk> token"""

    corpus = []
    for sentence in data[col].iteritems():
        word_list = sentence[1].split(" ")
        #[word if word not in rare else '<unk>' for word in sentence.split(" ")]
        #c = [[w for w in s.split(" ")] for s in data[col]]

        #word_list=replace_with_unks(word_list)
        corpus.append(word_list)
        #If not in infrequent then...

    # threshold = 20
    # counter = Counter(itertools.chain(*corpus))
    # rare = [word for word, freq in counter.items() if freq < threshold]
    #
    # corpus_final = []
    # for word_list in corpus:
    #     replaced_list = [word if word not in rare else '<unk>' for word in word_list]
    #     corpus_final.append(replaced_list)
    #
    # return corpus_final
    return corpus

    #rare = [word for word, freq in counter.items() if freq < threshold]
    #corpus = [[word if word not in rare else '<unk>' for word in sentence.split(" ")] for sentence in data[col]] # SLOW!!!


def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

#tsne_plot(model)

# Create an option for saving the word embeddings.
# Create padding and deleting functions
# Lookup word in embedding (similarity?)

def vec_sequences(corpus, model):
    "Creates input for training"
    vec_sequences = []
    for sentence in corpus:
        vec_sequences.append([model[word] for word in sentence])
    return vec_sequences

def remove_infrequent(corpus, threshold):
    "Replaces rare words with UNK token"
    counter = Counter(itertools.chain(*corpus))

