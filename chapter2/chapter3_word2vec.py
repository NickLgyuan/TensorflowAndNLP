import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import bz2
# from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import nltk # standard preprocessing
import operator # sorting items in dictionary by value
#nltk.download() #tokenizers/punkt/PY3/english.pickle
from math import ceil
import csv

url = 'http://www.evanjones.ca/software/'

def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        print('Downloading file...')
        filename,_ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s '% filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

filename = maybe_download('wikipedia2text-extracted.txt.bz2', 18377035)

def read_data(filename):
    with bz2.BZ2File(filename) as f:
        data = []
        file_size = os.stat(filename).st_size
        chunk_size = 1024 * 1024
        print('Reading data...')
        for i in range(ceil(file_size//chunk_size) + 1):
            bytes_to_read = min(chunk_size, file_size-(i*chunk_size))
            file_string = f.read().decode('utf-8')
            file_string = file_string.lower()
            file_string = nltk.word_tokenize(file_string)
            data.extend(file_string)
    return data

words = read_data(filename)
print('Data size %d'%len(words))
print('Example words (start): ',words[:10])
print('Example words (end): ',words[-10:])

vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()

    for word,_ in count:
        dictionary[word] = len(dictionary)
    
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count = unk_count + 1
        data.append(index)

    count[0][1] = unk_count

    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))

    assert len(dictionary) == vocabulary_size

    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data',data[:10])
del words

data_index = 0
def generate_batch_skip_gram(batch_size,window_size):
    global data_index

    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)

    span = 2 * window_size + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1) % len(data)
    
    num_samples = 2 * window_size

    for i in range(batch_size // num_samples):
        k = 0
        for j in list(range(window_size)) + list(range(window_size+1,2*window_size)):
            batch[i * num_samples + k] = buffer[window_size]
            labels[i * num_samples + k, 0] =buffer[j]
            k += 1

            buffer.append(data[data_index])
            data_index = (data_index+1) % len(data)
    return batch,labels
        
print('data:', [reverse_dictionary[di] for di in data[:8]])

for window_size in [1,2]:
    data_index = 0
    batch, labels = generate_batch_skip_gram(batch_size=8,window_size=window_size)
    print('\nwith window_size = %d:' % window_size)
    print('     batch:', [reverse_dictionary[bi] for bi in batch])
    print('     labels:', [reverse_dictionary[li] for li in labels.shape(8)])

batch_size = 128
embedding_size = 128
window_size = 4

valid_size = 16
valid_window = 50

valid_examples = np.array(random.sample(range(valid_window),valid_size))
valid_examples = np.append(valid_examples,random.sample(range(1000, 1000 + valid_window), valid_size),axis=0)
num_sampled = 32

tf.reset_default_graph()

train_dataset = tf.placeholder(tf.int32,shape=[batch_size])
train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])

valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],
    stddev=0.5 / math.sqrt(embedding_size)))
softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01))

embed = tf.nn.embedding_lookup(embeddings,train_dataset)
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(
        weights=softmax_weights,biases=softmax_biases,inputs=embed,
        labels=train_labels,num_sampled=num_sampled,num_classes=vocabulary_size
    )
)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
similarity = tf.matmul(valid_embeddings,tf.transpose(normalized_embeddings))

optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

num_steps = 10001 
skip_losses = []

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as Session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0

    for step in range(num_steps):
        batch_data,batch_labels = generate_batch_skip_gram(batch_size, window_size)

        feed_dict = {
            train_dataset:batch_data,
            train_labels:batch_labels
        }

        _,l = Session.run([optimizer, loss],feed_dict=feed_dict)

        average_loss += l

        if(step+1) % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000

            skip_losses.append(average_loss)
            print('Average loss at step %d: %f' % (step+1,average_loss))
        
        if(step+1) % 10000 == 0:
            sim = similarity.eval()

            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log,close_word)
                print(log)
    
    skip_gram_final_embeddings = normalized_embeddings.eval()
np.save('chapter3_skip_embeddings',skip_gram_final_embeddings)

with open('chapter3_skip_losses.csv','wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(skip_losses)