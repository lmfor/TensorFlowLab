# Package Imports
import tensorflow as tf
import numpy as np
import pandas as pd

# Method Imports
from get_data import get_csv

"""
TASKS:

1. Creat tensors with explicit dtype (tf.string, tf.int32)         COMPLETE
2. Convert between NumPy and TF                                    COMPLETE
3. Demonstrate broadcasting with a few examples                    COMPLETE (focus)
4. Show the difference between:
    a. tf.constant(...)
    b. tf.Variable(...)                                            COMPLETE
5. Update a variable via assign, assign_add                        COMPLETE (focus)
6. Print .shape, .dtype, .device                                   COMPLETE

Success Check:
I can explain in one sentence: "Tensors are values, Variables are state you can mutate"

"""

# Prep Data
products = get_csv("..\\data\\products-1000.csv").dropna(axis=1)

"""                                                 
============================ 1 & 2 ============================
"""

# NumPy Array
prod_np = products.to_numpy() # (1000, 13) | (row, col)
prod_price_np = prod_np[:, 7]
prod_currency_np = prod_np[:, 6]

# Convert to tf
prices_tf = tf.Variable(initial_value=prod_price_np, dtype=tf.int32)
currency_tf = tf.constant(value=prod_currency_np, dtype=tf.string)

"""                                                 
============================  3 & 4 ============================
"""

# Broadcasting

#1. Scalar -> Vector
tax = tf.constant(1.08, dtype=tf.float32)
prices_f = tf.cast(prices_tf, tf.float32) # cast prices to dtype tf.float32
prices_with_tax = prices_f * tax
print(prices_with_tax[:20])

#2. Vector + Vector
discount = tf.ones(shape=(1000,), dtype=tf.float32) * 0.9 # 10% off
discounted = prices_f * discount
print(discounted[:20])

#3. Column Vector 
prices_col = tf.reshape(prices_f, shape=(1000,1)) # [ [p0], [p1], [p2], ...]
fees = tf.constant(value=[0.0, 1.5, 3.0], dtype=tf.float32) # shape=(3,)
result = prices_col + fees   
print(result[:20])
# prices_col : (1000, 1)
# fees       :       (3)
#[
# [p0 + 0.0, p0 + 1.5, p0 + 3.0],
# [p1 + 0.0, p1 + 1.5, p1 + 3.0],
# ...
# [p999 + 0.0, p999 + 1.5, p999 + 3.0]
#]


#4. Boolean Mask
expensive = prices_f > 100 # type: ignore
prices_flagged = tf.where(condition=expensive, x=prices_f, y=0.0)
print(prices_flagged[:20])

"""                                                 
============================   5   ============================
"""


v_tf = tf.Variable(initial_value=5)
v_tf.assign(10)
print(v_tf.numpy()) # type: ignore

v_tf.assign_add(15)
print(v_tf.numpy()) # type: ignore

prices = tf.Variable(initial_value=prices_f)
# 10% increase
# increase = 1.10
# prices.assign_add(prices*increase) # type:ignore
# print(prices.numpy()[:20]) # type: ignore

# Conditional Update
mask = prices < 100 # type: ignore

# tf.where(condition, x, y)
#  x if condition else y
prices.assign(
    tf.where(condition=mask, x=prices*2, y=prices) # type: ignore
)

print(prices.numpy()[:20]) # type: ignore




