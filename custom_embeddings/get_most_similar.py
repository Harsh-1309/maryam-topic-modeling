import numpy as np
import json
from numpy import dot
from numpy.linalg import norm

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

def cosine_similarity(vec1,vec2):
    vec1=np.array(vec1)
    vec2=np.array(vec2)
    cos_sim = dot(vec1, vec2)/(norm(vec1)*norm(vec2))
    return cos_sim

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    
    query_vector = vectors[word_index]
    # print(type(query_vector))
    # vectors.pop(word_index)
    for index, vector in enumerate(vectors):
        if euclidean_dist(vectors[vector], query_vector) < min_dist and not np.array_equal(vector, word_index):
            min_dist = euclidean_dist(vectors[vector], query_vector)
            min_index = index
    return min_index


def find_closest_cs(word_index, vectors):
    query_vector = vectors[word_index]
    # vectors.pop(word_index)

    max_sim=0
    max_index=-1

    for index,vector in enumerate(vectors):
        if cosine_similarity(vectors[vector],query_vector) > max_sim and not np.array_equal(vector, word_index):
            max_sim = cosine_similarity(vectors[vector], query_vector)
            max_index = index
    return max_index

print(int2word[str(find_closest('royal', vectors))])
print(int2word[str(find_closest_cs('royal', vectors))])