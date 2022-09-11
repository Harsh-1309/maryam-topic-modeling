from cgi import test
from functools import partial
from optparse import Values
from queue import Empty
import numpy as np
import json
from constants import *
from tqdm import tqdm

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
## Now continue with getting context from the context map

f = open(CONTEXT_MAP_PATH,"r")
data = dict(json.load(f))

padding_onehot=np.zeros(vocab_size)
def add_padding():
    return np.zeros(3)

def check_left_pad(word_ind,len):
    #return true if padding not required
    if word_ind >= WINDOW:
        return True
    else:
        return False

def check_right_pad(word_ind,len):
    #return true if padding not required
    if (len -1 - word_ind) >= WINDOW:
        return True
    else:
        return False

def left_pad_count(word_ind,len):
    return WINDOW-word_ind

def right_pad_count(word_ind,len):
    return WINDOW - (len -1 - word_ind)


def indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

x=[]
y=[]


with open(DATASET_FILE,"r") as f:
    for line in f:
        line=line.lower().strip().split()
        for word in line:
            context_=data[word]

            # if words are repeated multiple times in same line
            full_dataset_context=[]
            word_index=indices(line,word)
            line_len=len(line)

            for wi in word_index:
                full_context=[]
                if check_left_pad(wi,line_len)==True and check_right_pad(wi,line_len)==True:
                    for i in range(WINDOW):
                        full_context.append(line[wi-i-1]) #left context
                        full_context.append(line[wi+i+1]) # followed by right context
                    full_dataset_context.append(full_context)
                    # print(full_dataset_context, " if se aaya")
                else:   

                    left_check=check_left_pad(wi,line_len)
                    right_check=check_right_pad(wi,line_len)

                    if left_check == True:
                        # rpc= required pad count
                        rpc= right_pad_count(wi,line_len)
                        avail_words=WINDOW-rpc
                        for i in range(WINDOW):
                            full_context.append(line[wi-i-1])

                            if avail_words !=0:
                                full_context.append(line[wi+i+1])
                                avail_words-=1
                            else:
                                full_context.append(PAD_STRING)
                        # print(full_context, " else if se aaya")

                    elif right_check == True:
                        # rpc= required pad count
                        rpc= left_pad_count(wi,line_len)
                        avail_words=WINDOW-rpc
                        for i in range(WINDOW):
                            if avail_words !=0:
                                full_context.append(line[wi-i-1])
                                avail_words-=1
                            else:
                                full_context.append(PAD_STRING)

                            full_context.append(line[wi+i+1])


                        full_dataset_context.append(full_context)
                        # print(full_dataset_context, " if elif aaya")

                    else:
                        rpc_right= right_pad_count(wi,line_len)
                        rpc_left= left_pad_count(wi,line_len)
                        avail_words_left=WINDOW-rpc_left
                        avail_words_right=WINDOW-rpc_right

                        for i in range(WINDOW):
                            if avail_words_left !=0:
                                full_context.append(line[wi-i-1])
                                avail_words_left-=1
                            else:
                                full_context.append(PAD_STRING)

                            if avail_words_right !=0:
                                full_context.append(line[wi+i+1])
                                avail_words_right-=1
                            else:
                                full_context.append(PAD_STRING)

                        full_dataset_context.append(full_context)
                        # print(full_dataset_context, " if se aaya")

            x.extend(full_dataset_context)
            for _ in range(len(full_dataset_context)):
                y.append(word)
        


print(x)
print(y)
 
# replace the context strings with corresponding context matrix

x_context=[]
y_context=[]

for index,value in tqdm(enumerate(x)):
    temp=[]
    for k in value:
        if k == PAD_STRING:
            temp.append([0,0,0])
        else:
            if k != y[index]:
                temp.append(data[y[index]][k])
            # word in self context is still an issue
            else:
                temp.append([0,1,0])
    x_context.append(temp)
    y_context.append(y[index])


print(x_context)
print(y_context)

x_onehot=[]
y_onehot=[]

print(one_hot)

for index,value in enumerate(x):
    temp=[]
    for k in value:
        if k == PAD_STRING:
            temp.append(list(padding_onehot))
        else:
            temp.append(list(one_hot[k]))

    x_onehot.append(temp)
    y_onehot.append(list(one_hot[y[index]]))

print(x_onehot)
print(y_onehot)