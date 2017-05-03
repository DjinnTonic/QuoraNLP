import pandas as pd
import f_embed as emd
from gensim.models import word2vec

initial_sample = 100000


data = pd.read_csv("train.csv").sample(initial_sample)
data = emd.clean_dataframe(data)

corpus1 = emd.build_corpus(data, 'question1')
model1 = word2vec.Word2Vec(corpus1, size=100, window=50, min_count=200, workers=4)

corpus2 = emd.build_corpus(data, 'question2')
model2 = word2vec.Word2Vec(corpus2, size=100, window=50, min_count=200, workers=4)

emd.tsne_plot(model1)
emd.tsne_plot(model2)

print(len(model1.wv.vocab.keys()))
print(len(model2.wv.vocab.keys()))