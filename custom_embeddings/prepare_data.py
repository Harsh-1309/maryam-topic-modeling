import numpy as np
import json
from constants import *
from tqdm import tqdm

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

def convert_to_onehot(data_index,vocab_size):
    temp_array=np.zeros(vocab_size)
    temp_array[data_index]=1
    return temp_array

for key,val in word2int.items():
    one_hot[key]=convert_to_onehot(val,vocab_size)

# one hot encoded data achieved till now.
# combine this with the [dist,freq,prob] to get final data
# Now continue with getting context from the context map

f = open(CONTEXT_MAP_PATH,"r")
data = dict(json.load(f))

# define one hot encoded padding vector
padding_onehot=np.zeros(vocab_size)
def add_padding():
    return np.zeros(3)

#return true if left padding is not required
def check_left_pad(word_ind,len):
    if word_ind >= WINDOW:
        return True
    else:
        return False

#return true if right padding is not required
def check_right_pad(word_ind,len):
    if (len -1 - word_ind) >= WINDOW:
        return True
    else:
        return False

# returns padding count if required 
def left_pad_count(word_ind,len):
    return WINDOW-word_ind

def right_pad_count(word_ind,len):
    return WINDOW - (len -1 - word_ind)

# get all indices
def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

x=[]
y=[]

with open(DATASET_FILE,"r") as f:
    # iterate line by line through dataset
    for line in tqdm(f):
        line=line.lower().strip().split()
        for word in line:
            context_=data[word]

            # if words are repeated multiple times in same line
            full_dataset_context=[]
            # get all indices. Same word multiple times also works
            word_index=indices(line,word)
            line_len=len(line)

            # iterate for each retrived index 
            for wi in word_index:
                full_context=[]

                # enter if if no padding required on both sides
                if check_left_pad(wi,line_len)==True and check_right_pad(wi,line_len)==True:
                    for i in range(WINDOW):
                        full_context.append(line[wi-i-1]) #left context
                        full_context.append(line[wi+i+1]) # followed by right context
                    full_dataset_context.append(full_context)

                # enter this block when padding required on atleast 1 side
                else:   

                    left_check=check_left_pad(wi,line_len)
                    right_check=check_right_pad(wi,line_len)

                    # enter if padding on right is required
                    if left_check == True:
                        # rpc= required pad count
                        rpc= right_pad_count(wi,line_len)
                        avail_words=WINDOW-rpc
                        for i in range(WINDOW):
                            full_context.append(line[wi-i-1]) #left context

                            if avail_words !=0:
                                full_context.append(line[wi+i+1]) #right context
                                avail_words-=1
                            else:
                                full_context.append(PAD_STRING)

                    # enter if padding on left is required
                    elif right_check == True:
                        # rpc= required pad count
                        rpc= left_pad_count(wi,line_len)
                        avail_words=WINDOW-rpc
                        for i in range(WINDOW):
                            if avail_words !=0:
                                full_context.append(line[wi-i-1]) #left context
                                avail_words-=1
                            else:
                                full_context.append(PAD_STRING)

                            full_context.append(line[wi+i+1]) #right context


                        full_dataset_context.append(full_context)

                    # enter if padding required on both sides
                    else:
                        rpc_right= right_pad_count(wi,line_len)
                        rpc_left= left_pad_count(wi,line_len)
                        avail_words_left=WINDOW-rpc_left
                        avail_words_right=WINDOW-rpc_right

                        for i in range(WINDOW):
                            if avail_words_left !=0:
                                full_context.append(line[wi-i-1]) #left context
                                avail_words_left-=1
                            else:
                                full_context.append(PAD_STRING)

                            if avail_words_right !=0:
                                full_context.append(line[wi+i+1]) #right context
                                avail_words_right-=1
                            else:
                                full_context.append(PAD_STRING)

                        full_dataset_context.append(full_context)

            # extend by context of all words
            # each context will have a fixed size of WINDOW*2 
            x.extend(full_dataset_context)
            for _ in range(len(full_dataset_context)):
                y.append(word)
        
 
# replace the context strings with corresponding context matrix
# and replace with one hot encoded vectors 
x_context=[]
y_context=[]

x_onehot=[]
y_onehot=[]

for index,value in tqdm(enumerate(x)):
    temp1=[]
    temp2=[]
    for k in value:
        if k == PAD_STRING:
            temp1.append([0,0,0])
            temp2.append(list(padding_onehot))
        else:
            temp2.append(list(one_hot[k]))
            if k != y[index]:
                temp1.append(data[y[index]][k])
            # word in self context is still an issue!!
            else:
                temp1.append([0,1,0])
    x_context.append(temp1)
    y_context.append(y[index])

    x_onehot.append(temp2)
    y_onehot.append(list(one_hot[y[index]]))

# multiply one hot array and context map array
# to get final data
x_train=[]
y_train=np.asarray(y_onehot,dtype="float64")

# multiply element-wise
for oneh,con in tqdm(zip(x_onehot,x_context)):
    temp=[]
    for j,k in zip(oneh,con):
        multiplied_data=np.multiply.outer(np.array(j),np.array(k))
        temp.append(multiplied_data)

    # array flattened here to reduce complexity at this point
    # ideally, it shouldn't be and will be changed later on to
    # retain full shape for no distortion of information
    x_train.append(np.array(temp).flatten())
        
x_train=np.asarray(x_train,dtype="float64")

# save the data arrays to be used for training later on
np.save("data/x_train.npy",x_train)
np.save("data/y_train.npy",y_train)

# save word2int to get final word embeddings after the training
with open("data/word2int.json","w") as f:
    json.dump(word2int,f)