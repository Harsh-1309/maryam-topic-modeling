from calendar import EPOCH
import os
from constants import *
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import Input
import numpy as np
import json

# x = Input(shape=(None,WINDOW*2,VOCAB_SIZE,3))
# y = Input(shape=(None,VOCAB_SIZE))


EMB_DIM=DIMENSION
x_train=np.load("data/x_train.npy")
y_train=np.load("data/y_train.npy")

print("loaded arrays\n")

f = open("data/word2int.json","r")
word2int = dict(json.load(f))
f.close()

#initialise with random weights
# w1=tf.Variable(tf.random.normal([WINDOW*2,VOCAB_SIZE,3,EMB_DIM]))
# b1=tf.Variable(tf.random.normal([EMB_DIM]))

opt_adam=tf.optimizers.Adam()
opt_sgd=tf.optimizers.SGD()
EPOCHS=200
FINAL_DIM=VOCAB_SIZE*WINDOW*2*3

def train(x_train,y_train,optimizer=opt_sgd):
    w1=tf.Variable(tf.random.normal([FINAL_DIM,EMB_DIM]))
    b1=tf.Variable(tf.random.normal([EMB_DIM]))
    w2=tf.Variable(tf.random.normal([EMB_DIM,VOCAB_SIZE]))
    b2=tf.Variable(tf.random.normal([VOCAB_SIZE]))

    for _ in range(EPOCHS):
        with tf.GradientTape(persistent=True) as t:
            hidden_layer=tf.add(tf.matmul(x_train,w1),b1)
            output_layer = tf.nn.softmax(tf.add( tf.matmul(hidden_layer,w2),b2))
            # print(output_layer)
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

print(f"Training complete for {EPOCHS} epochs gg")

print("\nTest embeddings\n")

print(get_vector(w1,b1,word2int["used"]))

"""
Sample output of embedding for word "used":

tf.Tensor(
[ 3.307526    1.2745522   0.6221365   1.0608096  -1.3911397  -2.7389529
 -0.03785023 -1.6968094   0.61216784  0.995255  ], shape=(10,), dtype=float32)

Shape will be dimension dependent
"""