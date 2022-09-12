"""
These constants will be used throughout the code base
"""

DATASET_FILE ="data/text7"
PAD_STRING = "%s"
WINDOW=3 #make it 3
CONTEXT_MAP_PATH = "data/context_map_text7.json"
WINDOW_FLOAT = 3.0
MAX_RATE = 8.0
DIMENSION = 100 # initially dim was 100
SAMPLE_ACCURACY = 4000
EPOCHS=5

WORDS=[]
CONTEXT={} # This dict will store context map for entire dataset
# VOCAB_SIZE=7 # was 14 for text6

VOCAB_SIZE=627
FINAL_DIM=VOCAB_SIZE*WINDOW*2*3