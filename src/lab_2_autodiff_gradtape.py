import tensorflow as tf
import numpy as np
import pandas as pd
# import math

"""
TASKS:

1. Implement a scalr loss: L = (w*x + b -y)^2
2. Compute gradients dL/dw, dL/db
3. Show what happens if:
    a. I use tf.constant instead of tf.Variable
    b. I froget tape.watch(...) on a Tensor
4. Compute gradients fora  vectorized batch version
5. Print gradient norms and detect exploding gradients

Success check:
I can debug why a gradient is None without guessing.
"""

"""
==========================    1    =========================
"""

tf.random.set_seed(1333)
np.random.seed(1333)

shoes = tf.constant([10, 11, 12, 13, 14], dtype=tf.float32)
height = tf.constant([160, 170, 180, 190, 200], dtype=tf.float32)
# shoes = height * w + b

w = tf.Variable(tf.random.normal([1,]),)
b = tf.Variable(tf.random.normal([1,]),)

print(f"w: {w.numpy()}\nb: {b.numpy()}") # type: ignore

with tf.GradientTape() as tape:
    y_hat = height * w + b # type: ignore
    loss = tf.reduce_mean(tf.square(shoes - y_hat))

#                     func, respect to __
dw, db = tape.gradient(loss, [w,b]) # type: ignore
#x = tf.Variable(3., dtype=tf.float32)
#with tf.GradientTape() as tape:
#    y = x ** 2 # type: ignore
    
##               (func, respect to __)
#dy_dx = tape.gradient(y, [x])

"""
==========================    2    =========================
"""


