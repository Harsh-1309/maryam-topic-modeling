import numpy as np
import json

with open("saved_embeddings/generated_embeddings.json","r") as f:
    embeddings=dict(json.load(f))

vectors=embeddings

f = open("data/word2int.json","r")
word2int = dict(json.load(f))
f.close()

f = open("data/int2word.json","r")
int2word = dict(json.load(f))
f.close()

def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((np.array(vec1)-np.array(vec2))**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    
    query_vector = vectors[word_index]
    # print(type(query_vector))
    vectors.pop(word_index)
    for index, vector in enumerate(vectors):
        if euclidean_dist(vectors[vector], query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vectors[vector], query_vector)
            min_index = index
    return min_index

print(int2word[str(find_closest('seven', vectors))])