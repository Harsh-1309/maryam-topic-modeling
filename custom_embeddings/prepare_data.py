import numpy as np
import json

DATASET_FILE ="data/text6"
PAD_STRING = "%s"
WINDOW=3

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

## Now continue with getting context from the context map

padding_onehot=np.zeros(vocab_size)

f = open("data/context_map_text6.json","r")
data = json.load(f)
data = dict(data)

# print(data)

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

with open(DATASET_FILE,"r") as f:
    for line in f:
        line=line.lower().strip().split()
        for word in line:
            context_=data[word]
            print("context starts\n")
            print(context_,len(context_))
            # if words are repeated multiple times in same line
            full_context=[]
            word_index=indices(line,word)
            line_len=len(line)

            # if len(context_)>=WINDOW*2:
            #     print(line)
            #     print(word_index)

            for wi in word_index:
                if check_left_pad(wi,line_len)==True and check_right_pad(wi,line_len)==True:
                    for i in range(WINDOW):
                        if len(full_context) <6:
                            full_context.append(line[wi-i-1]) #left context
                            full_context.append(line[wi+i+1]) # followed by right context
                    print(full_context, " if se aaya")
                else:   

                    left_check=check_left_pad(wi,line_len)
                    right_check=check_right_pad(wi,line_len)

                    if left_check == True:
                        # rpc= required pad count
                        rpc= right_pad_count(wi,line_len)
                        avail_words=WINDOW-rpc
                        print("right mai chaiye ", rpc, avail_words)
                        for i in range(WINDOW):
                            if len(full_context) <6:
                                full_context.append(line[wi-i-1])
                            if avail_words !=0:
                                if len(full_context) <6:
                                    full_context.append(line[wi+i+1])
                                avail_words-=1
                            else:
                                if len(full_context) <6:
                                    full_context.append(PAD_STRING)
                        print(full_context, " else if se aaya")

                    elif right_check == True:
                        # rpc= required pad count
                        rpc= left_pad_count(wi,line_len)
                        avail_words=WINDOW-rpc
                        print("left mai chaiye ",rpc,avail_words)
                        for i in range(WINDOW):
                            if avail_words !=0:
                                if len(full_context) <6:
                                    full_context.append(line[wi-i-1])
                                avail_words-=1
                            else:
                                if len(full_context) <6:
                                    full_context.append(PAD_STRING)

                            if len(full_context) <6:
                                full_context.append(line[wi+i+1])
                        print(full_context, " else elif se aaya")

                    else:
                        rpc_right= right_pad_count(wi,line_len)
                        rpc_left= left_pad_count(wi,line_len)
                        avail_words_left=WINDOW-rpc_left
                        avail_words_right=WINDOW-rpc_right

                        for i in range(WINDOW):
                            if avail_words_left !=0:
                                if len(full_context) <6:
                                    full_context.append(line[wi-i-1])
                                avail_words_left-=1
                            else:
                                if len(full_context) <6:
                                    full_context.append(PAD_STRING)

                            if avail_words_right !=0:
                                if len(full_context) <6:
                                    full_context.append(line[wi+i+1])
                                avail_words_right-=1
                            else:
                                if len(full_context) <6:
                                    full_context.append(PAD_STRING)

                        print(full_context, " else else se aaya")
                        # print("PADDING ADD KAR else wala")
                # print("YET TO DO MULTIPLE CONTEXT")

            # else:
            #     for wi in word_index:
            #         left_check=check_left_pad(wi,line_len)
            #         right_check=check_right_pad(wi,line_len)
            #         if left_check == True:
            #             # rpc= required pad count
            #             rpc= right_pad_count(wi,line_len)
            #             avail_words=WINDOW-rpc
            #             print("right mai chaiye ", rpc, avail_words)
            #             for i in range(WINDOW):
            #                 full_context.append(line[wi-i-1])
            #                 if avail_words !=0:
            #                     full_context.append(line[wi+i+1])
            #                     avail_words-=1
            #                 else:
            #                     full_context.append(PAD_STRING)
            #             print(full_context)

            #         elif right_check == True:
            #             # rpc= required pad count
            #             rpc= left_pad_count(wi,line_len)
            #             avail_words=WINDOW-rpc
            #             print("left mai chaiye ",rpc,avail_words)
            #             for i in range(WINDOW):
            #                 if avail_words !=0:
            #                     full_context.append(line[wi-i-1])
            #                     avail_words-=1
            #                 else:
            #                     full_context.append(PAD_STRING)
            #                 full_context.append(line[wi+i+1])
            #             print(full_context)

            #         else:
            #             rpc_right= right_pad_count(wi,line_len)
            #             rpc_left= left_pad_count(wi,line_len)
            #             avail_words_left=WINDOW-rpc_left
            #             avail_words_right=WINDOW-rpc_right

            #             for i in range(WINDOW):
            #                 if avail_words_left !=0:
            #                     full_context.append(line[wi-i-1])
            #                     avail_words_left-=1
            #                 else:
            #                     full_context.append(PAD_STRING)

            #                 if avail_words_right !=0:
            #                     full_context.append(line[wi+i+1])
            #                     avail_words_right-=1
            #                 else:
            #                     full_context.append(PAD_STRING)

            #             print(full_context)
                            
                # print("YAHA BHI PADDING CHAIYE 1")

