from calendar import EPOCH
import os
from constants import *
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import Input
import numpy as np
import json
import pickle

EMB_DIM=DIMENSION
x_train=np.load("data/x_train.npy")
y_train=np.load("data/y_train.npy")
print("loaded arrays")

f = open("data/word2int.json","r")
word2int = dict(json.load(f))
f.close()

opt_adam=tf.optimizers.Adam()
opt_sgd=tf.optimizers.SGD()

EPOCHS=200
FINAL_DIM=VOCAB_SIZE*WINDOW*2*3

"""
DO NOT CHANGE DTYPE TO FLOAT32. CRASHES RANDOMLY.
"""
def train(x_train,y_train,optimizer=opt_sgd):
    w1=tf.Variable(tf.random.normal([FINAL_DIM,EMB_DIM],dtype="float64"),dtype="float64")
    b1=tf.Variable(tf.random.normal([EMB_DIM],dtype="float64"),dtype="float64")
    w2=tf.Variable(tf.random.normal([EMB_DIM,VOCAB_SIZE],dtype="float64"),dtype="float64")
    b2=tf.Variable(tf.random.normal([VOCAB_SIZE],dtype="float64"),dtype="float64")

    for _ in range(EPOCHS):
        with tf.GradientTape() as t:
            hidden_layer=tf.add(tf.matmul(x_train,w1),b1)
            output_layer = tf.nn.softmax(tf.add( tf.matmul(hidden_layer,w2),b2))
            cross_entropy_loss = tf.reduce_mean(-tf.math.reduce_sum(y_train * tf.math.log(output_layer), axis=[1]))

            grads = t.gradient(cross_entropy_loss, [w1,b1,w2,b2])
            optimizer.apply_gradients(zip(grads,[w1,b1,w2,b2]))
            if(_ % 5 == 0):
                print("loss: ", cross_entropy_loss)

    return w1,b1

def get_vector(w1,b1,word_idx):
    return(w1+b1)[word_idx]

print("start training\n")
w1,b1=train(x_train,y_train)
print(f"\nTraining complete with {EPOCHS} epochs\n")

with open("saved_weights/w1","wb") as f:
    pickle.dump(w1,f)

with open("saved_weights/b1","wb") as f:
    pickle.dump(b1,f)

print("weights abd biases saved at 'saved_weights/'")

"""
Sample output of embedding for word "used":

tf.Tensor(
[ 3.307526    1.2745522   0.6221365   1.0608096  -1.3911397  -2.7389529
 -0.03785023 -1.6968094   0.61216784  0.995255  ], shape=(10,), dtype=float32)

Shape will be dimension dependent
"""

#save all word embeddings
all_words=list(word2int.keys())
word_embeddings={}

for word in all_words:
    word_embeddings[word]=list(np.asarray(get_vector(w1,b1,word2int[word])))

with open("saved_embeddings/generated_embeddings.json","w") as f:
    json.dump(word_embeddings,f)

print("all word embeddings saved at 'saved_embeddings/generated_embeddings.json")
