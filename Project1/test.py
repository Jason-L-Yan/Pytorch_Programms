import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers.core import Dense, Activation, Dropout
# from tensorflow.keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.image as processimage

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
