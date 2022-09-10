import numpy as np
import json

DATASET_FILE ="data/text6"

words=[]

with open(DATASET_FILE,"r") as f:
    for line in f:
        line=line.lower().strip().split()
        for word in line:
            words.append(word)

#using this and not set to retain order.
#set distorts ordering with every run, might result in expected results
words=list(dict.fromkeys(words))
vocab_size=len(words)

word2int = {}
int2word = {}
one_hot = {}

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

def convert_to_onehot(data_index,vocab_size):
    temp_array=np.zeros(vocab_size)
    temp_array[data_index]=1
    return temp_array

for key,val in word2int.items():
    one_hot[key]=convert_to_onehot(val,vocab_size)

# one hot encoded data achieved till now.
# combine this with the [dist,freq,prob] to get final data
# print(words)
# print(word2int)
# print(one_hot)

