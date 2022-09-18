import numpy as np
import json
from constants import *
from tqdm import tqdm
import gc

# create a list for all the words int he datset
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


#vocab size will be updated based on dataset used
with open("constants.py","a") as const:
    const.write(f"\nVOCAB_SIZE={vocab_size}")

#These dictionaries will be helpful later on. 
#Word2int maps words as keys againts int as value
#int2word does exactly opposite of this
#onehot has one hot encoded vectors mapped against words as keys

word2int = {}
int2word = {}
one_hot = {}

for i,word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

del words
gc.collect()

def convert_to_onehot(data_index,vocab_size):
    temp_array=np.zeros(vocab_size)
    temp_array[data_index]=1
    return temp_array

for key,val in word2int.items():
    one_hot[key]=convert_to_onehot(val,vocab_size)
gc.collect()
# one hot encoded data 
# achieved till now.
# combine this with the [dist,freq,prob] to get final data
# Now continue with getting context from the context map

f = open(CONTEXT_MAP_PATH,"r")
data = dict(json.load(f))
f.close()

def modify_vector(base, multiplying_factor):
    base=np.array(base)
    multiplying_factor=np.array(multiplying_factor)
    if multiplying_factor[0]>0:
        hmm= np.divide(np.multiply(base,multiplying_factor[1]),multiplying_factor[0])
    else:
        hmm= np.divide(np.multiply(base,multiplying_factor[1]),1)

    return hmm

x=[]
y=[]

with open(DATASET_FILE,"r") as f:
    # iterate line by line through dataset
    for line in tqdm(f):
        line=line.lower().strip().split()
        for word in range(len(line)):
            context_=data[line[word]]

            temp_x=[]
            temp_y=[]

            for w in range(1,WINDOW+1):
                try:

                    temp_x.append(modify_vector(one_hot[line[word+w]],context_[line[word+w]]))
                    temp_y.append(one_hot[line[word]])
                except:
                    pass
                try:
                    if word-w>=0:
                        temp_x.append(modify_vector(one_hot[line[word-w]],context_[line[word-w]]))
                        temp_y.append(one_hot[line[word]])
                except:
                    pass

            x.extend(temp_x)
            y.extend(temp_y)

        gc.collect()

# save word2int to get final word embeddings after the training
with open("data/word2int.json","w") as f:
    json.dump(word2int,f)

with open("data/int2word.json","w") as f:
    json.dump(int2word,f)

x_train=np.asarray(x,dtype="float64")
y_train=np.asarray(y,dtype="float64")

# save the data arrays to be used for training later on
np.save("data/x_train.npy",x_train)
np.save("data/y_train.npy",y_train)