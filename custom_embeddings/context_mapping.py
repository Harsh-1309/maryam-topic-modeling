import json
import pprint
from constants import *
from tqdm import tqdm

def counter_update(list,newlist):
    for _,x in enumerate(newlist):
        # value for key x same in both list and newlist then go into if
        # else add to list. Both list and newlist are dictionaries.
        if x in list.keys():
            # update distance using context window
            list[x][0] = WINDOW_FLOAT - (list[x][0]+newlist[x][0])/2
            # update frequency count
            list[x][1] += newlist[x][1]
        else:
            list[x]=newlist[x]
    return list


def makeContext():
    window={}
    f=open(DATASET_FILE,"r")
    # read and iterate through each line of the dataset
    for line in tqdm(f):
        line=line.lower().strip().split(" ") #Clean the data by splitting at spaces 
        doclen=len(line)

        #enumerate returns index, word from raw data array
        for i,k in enumerate(line):
            window={}
            # get context of size WINDOW, both prev and next from target word
            prev,next=line[max(i - WINDOW, 0):i], line[i + 1:min(1 + WINDOW + i, doclen)]
            #iterate in combined context array to return index, [prev] or [next]
            for mines,j in enumerate([prev,next]):
                pr=0
                nlen=len(j)

                for w,x in enumerate(j):
                    # for previous context
                    if mines == 0:
                        p = (nlen-w) -pr
                    #for next context
                    else:
                        p = (w+1) - pr
                    # if a word repeats in context window, then increament the counter of that word.
                    # distance will be updated as below later on
                    if x == k:
                        pr +=1
                    else:
                        # if word x already exists in window dict (word repeated)
                        if x in window.keys():
                            wx=window[x]
                            window[x][0]=WINDOW_FLOAT - ((wx[0]+p)/(wx[1]+1))
                        # else, create key and add the corresponding distance and frequency
                        else:
                            window[x]=[p,0,0]
                        # frequency counter update
                        window[x][1]+=1
            # trigger counter update function when a word is repated in context map.
            # do this to update the distance as per formula used above
            if k in CONTEXT.keys():
                CONTEXT[k]=counter_update(CONTEXT[k],window)
            # else, append the new word to context map
            else:
                WORDS.append(k)
                CONTEXT[k]=window
    f.close()

makeContext()
with open (f"{'/'.join(DATASET_FILE.split('/')[:-1])}/context_map_{DATASET_FILE.split('/')[-1]}.json","w") as cm:
    json.dump(CONTEXT,cm)
# pprint.pprint(CONTEXT)

"""
The output of this function will genrerate a context map, that is, distance of a word 
and its frequency wrt a word and for every word

Sample data:

anarchism originated as a term of abuse first
used early used working used a radicals including

Sample of the expected out (just a sample, not complete output):

{'a': {'abuse': [3, 1, 0],
       'anarchism': [3, 1, 0],
       'as': [1, 1, 0],
       'including': [2, 1, 0],
       'of': [2, 1, 0],
       'originated': [2, 1, 0],
       'radicals': [1, 1, 0],
       'term': [1, 1, 0],
       'used': [1.0, 2, 0],
       'working': [2, 1, 0]},

 'as': {'a': [1, 1, 0],
        'anarchism': [2, 1, 0],
        'of': [3, 1, 0],
        'originated': [1, 1, 0],
        'term': [2, 1, 0]},
 'early': {'used': [1.3333333333333333, 3, 0], 'working': [2, 1, 0]},
 
 'used': {'a': [1.5, 2, 0],
          'early': [0.25, 3, 0],
          'including': [3, 1, 0],
          'radicals': [2, 1, 0],
          'working': [2.25, 3, 0]},

 'working': {'a': [2, 1, 0],
             'early': [2, 1, 0],
             'radicals': [3, 1, 0],
             'used': [2.3333333333333335, 3, 0]}}



The last 0 from this context matrix will be replaced by probabilities soon.
"""
import gc
gc.collect()