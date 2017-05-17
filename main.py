import pandas as pd
import f_embed as emd
from gensim.models import word2vec

initial_sample = 100000
embedding_size = 200
max_length = 15 # Maximum sentence length
min_frequency = 20 # Minimum frequency of words in word2vec model

data = pd.read_csv("train.csv")#.sample(initial_sample)
data = emd.clean_dataframe(data)

corpus1 = emd.build_corpus(data, 'question1')
corpus1 = emd.replace_rare(corpus1, min_frequency)
model1 = word2vec.Word2Vec(corpus1, size=embedding_size, window=50, min_count=min_frequency, workers=8)

corpus2 = emd.build_corpus(data, 'question2')
corpus2 = emd.replace_rare(corpus2, min_frequency)
model2 = word2vec.Word2Vec(corpus2, size=embedding_size, window=50, min_count=min_frequency, workers=8)

#emd.tsne_plot(model1) # T-SNE plots (2-3 min with ~4000 words)
#emd.tsne_plot(model2)

print(len(model1.wv.vocab.keys()))
print(len(model2.wv.vocab.keys()))

# words1 = [k for k, v in model1.wv.vocab.items()]
# emb_matrix1 = [model1.wv[word] for word in words1] # Create embedding matrix
emb_matrix1 = model1.wv.syn0
# Here you can save the embedding using np.save for extraction later with TensorFlow/Keras

# train_data = tf.placeholder(shape=[batch_size, max_length, embedding_size], dtype=tf.float32)
import itertools
from collections import Counter
upd_counter = Counter(itertools.chain(*corpus1))