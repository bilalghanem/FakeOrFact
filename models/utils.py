import json
import os
import time
import re
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from collections import defaultdict

import keras
import keras.backend as K
import tensorflow as tf
from keras import metrics
from itertools import product
from functools import partial
from keras.losses import mean_squared_error

def w_categorical_crossentropy(y_true, y_pred):
  weights = np.array([[1., 5.],  # misclassify N -> Y
                      [10., 1.]])# misclassify Y -> N
  nb_cl = len(weights)
  final_mask = K.zeros_like(y_pred[:, 0])
  y_pred_max = K.max(y_pred, axis=1)
  y_pred_max = K.expand_dims(y_pred_max, 1)
  y_pred_max_mat = K.equal(y_pred, y_pred_max)
  for c_p, c_t in product(range(nb_cl), range(nb_cl)):
    final_mask += (
    K.cast(weights[c_t, c_p], K.floatx()) *
    K.cast(y_pred_max_mat[:, c_p],
           K.floatx()) *
    K.cast(y_true[:, c_t],K.floatx()))
  return K.categorical_crossentropy(y_pred, y_true) * final_mask

def precision(y_true, y_pred):
  y_true, y_pred = K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)
  y_true, y_pred = K.cast(y_true, 'float32'), K.cast(y_pred, 'float32')
  TP = K.sum(K.clip(y_true * y_pred, 0, 1)) # how many
  predicted_positives = K.sum(K.clip(y_pred, 0, 1))
  return TP / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
  y_true, y_pred = K.argmax(y_true, axis=1), K.argmax(y_pred, axis=1)
  y_true, y_pred = K.cast(y_true, 'float32'), K.cast(y_pred, 'float32')
  TP = K.sum(K.clip(y_true * y_pred, 0, 1))  # how many
  # TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  # possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  possible_positives = K.sum(K.clip(y_true, 0, 1))
  return TP / (possible_positives + K.epsilon())

def f1_score(y_true, y_pred):
  # If there are no true positives, fix the F score at 0 like sklearn.
  if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
      return 0
  p = precision(y_true, y_pred)
  r = recall(y_true, y_pred)
  fscore = 2 * (p * r) / (p + r + K.epsilon())
  return fscore



if __name__ == '__main__':
  pass

