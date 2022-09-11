# Custom Embeddings Model
### The following repo mainly contains of 2 parts:
- "word2vec/" includes basic retraining of word2vec architecure for custom data 
- "custom_embeddings/" has a novel architecure to generate word embeddings using a neural network, inspired by [this](https://github.com/callforpapers-source/inter-word-embedding) model.
### The model generates deep word embeddings based on the following factors combined:
- One hot encoding of the words.
- Full context of word, i.e, previous and forward.
- Distance of each context word from target/source word.

The more the information that can be passed as input, the better the vectors will be. This is because embeddings is all about representing the information distribution in the dataset.

The custom_embeddings directory has 4 scripts:
``` 
- constants.py
- context_mapping.py
- prepare_data.py
- embeddings.py 
```

- ` constants.py ` 
    - Has all the constants used in each of the scripts, including WINDOW, DATASET, VOCAB_SIZE or DIMENSION of the word vector.
    - Changes here will change the entire model once retrained or trained on custom data

- `context_mapping.py `
    - This gnerates the context in format [distance, word frequency, prob] with respect to each word in every line of the dataset
    - Repetition of context word, within or outside context window is also taken care of.
    - Distance and frequency will be updated as repetition occures, along with probability.
    - These contexts are saved in "custom_embeddings/data/" directory as json file.

    > Note: The probabilty of context has not been implemented yet, but will be done as soon as possible. Currently  "0" is used in place of probability for every word in the dataset. 

- `prepare_data.py`
    - This will generate one hot vectors for each word in the dataset and prepare the fixed context as well.
    - In case of no available context words, simple padding is used.
    - Finally, these 2 are multiplied to generate a final input vector or encoding which will be used for training.
    - This data is saved as numpy array in "custom_embeddings/data/" directory.

- `embeddings.py`
    - This has a simple implementation of a neural network with 1 input, 1 hidden and 1 output layer.
    - The weights between input and hidden layer give us the actual embeddings and hence is frequently termed as "embeddings layer" as well.
    - Layers can be increased or more specialised/complex layers can be added for better performance. 
    - The weights and biases of trained model and correspondingly generated word embeddings for each word in the dataset are saved in "saved_weights/" and "saved_embeddings/" directories respectively. 

###  This architecture follows simialr approach to how word2vec is trained, but uses more input information than word2vec. In theory it should perform better, but through testing and benchmarking is yet to be done.

### These embeddings can be used for any purpose, may it be classification or clustering or topic modelling. 
 