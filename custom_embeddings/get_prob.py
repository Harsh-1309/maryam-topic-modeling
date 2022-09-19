from constants import *
import json
from tqdm import tqdm

global_word_count={}

f = open(CONTEXT_MAP_PATH,"r")
data = dict(json.load(f))
f.close()

with open(DATASET_FILE,"r") as f:
    # iterate line by line through dataset
    for line in tqdm(f):
        words=line.lower().strip().split()
        for word in words:
            if word in global_word_count:
                global_word_count[word] += 1
            else:
                global_word_count[word] = 1

for i, out_key in enumerate(data):
    for j, in_key in enumerate(data[out_key]):
        data[out_key][in_key][2]=global_word_count[in_key]


with open (f"{'/'.join(DATASET_FILE.split('/')[:-1])}/context_map_{DATASET_FILE.split('/')[-1]}.json","w") as cm:
    json.dump(data,cm)