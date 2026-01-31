import tensorflow as tf
import numpy as np
import pandas as pd

# Index(['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play'], dtype='str')
golf_data = pd.read_csv("..\\data\\golf_data_1.csv")
golf_data = golf_data.to_numpy()  # For Series or DataFrame

# Task 1
tf_int = tf.constant(value=[1, 2, 3],dtype=tf.int32)
tf_float = tf.constant(value=[4.0, 5.0, 6.0], dtype=tf.float32)

# Task 2
np_int = tf_int.numpy() # type: ignore  # For Tensors
# print(type(np_int[0])) # type: ignore
tf_int_copy = tf.constant(value=np_int,dtype=tf.int32)
# print(tf_int_copy.dtype) # type: ignore

# Task 3
# print(golf_data[:, 1])
temp_tf = tf.constant(value=golf_data[:,1], dtype=tf.float32)
temp_tf = tf.reshape(tensor=temp_tf, shape=(14,1)) # type: ignore
colder_day_tf = tf.constant(-10.0, dtype=tf.float32)

# print(temp_tf)
cold_tf = temp_tf + colder_day_tf # type: ignore
# print(cold_tf)

# Task 4
one_tf = tf.constant(value=[6, 7, 8], dtype=tf.int32)
two_tf = tf.constant(value=[-1, -2, -3, -4, -5, -6], dtype=tf.int32) # (6,)
one_tf = tf.reshape(one_tf, shape=(3,1)) # (3,1)
# print(one_tf)
# print(two_tf)
three_tf = one_tf + two_tf 
# print(three_tf)

# Task 5
constant_tf = tf.constant(2.9, dtype=tf.float32)
variable_tf = tf.Variable(5.7, dtype=tf.float32)

# constant_tf.assign_add(9) # <<< THIS WON'T WORK BC ITS A CONSTANT TENSOR
variable_tf.assign_add(10)
# print(variable_tf.numpy()) # type: ignore

# Task 6
h = tf.Variable(initial_value=[1,4,7,10], dtype=tf.int32)
# print(h)
h.assign([2,5,8,11]) # Shapes need to match !! (4,)
# print(h)

# Task 7
h.assign_add([1, 2, 3, 4]) # Shapes need to match !! (4,)
# print(h)

# Task 8
print(h.shape)
print(h.device)
print(h.dtype)
