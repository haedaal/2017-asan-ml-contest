import dicom
import pylab
import numpy as np
from glob import glob
import os

import matplotlib
import matplotlib.pyplot as plt

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import itertools

virtual_max = 50

def preprocess(X):
    num = X.shape[0]
    X = X ** 0.5
    X = (X - virtual_max / 2) / virtual_max
    X = X.reshape(num, 80, 80, 1)
    return X

def getFromSlice(keyword):
    def get(slice):
        return slice[keyword]
    return get

def flatten(listOfList):
    return itertools.chain.from_iterable(listOfList)

# minimum length of xList should be >= batch_num
def symmetric_categorical_data_generator(xList, num_classes, batch_num, sym=False):
    assert(len(xList) == num_classes)
    bag = {}
    for i in range(num_classes):
        bag[i] = xList[i]
    while True:
        w = np.maximum((np.random.randn(num_classes) + 1.5), 0)
        nums = np.floor(w / w.sum() * batch_num)
        nums[np.random.randint(num_classes)] += (batch_num - nums.sum())
        X = []
        y = []
        for i in range(num_classes):
            X.append(xList[i][np.random.randint(0,xList[i].shape[0], int(nums[i]))])
            y.append(np.full(int(nums[i]), i))
        X = np.vstack(X)
        y = np.hstack(y)
        yield (X, to_categorical(y, num_classes))

def symmetric_data_generator(xList, yList, batch_num, sym=False, generate_all=False):
    num_category = len(xList)
    while True:
        w = np.maximum((np.random.randn(num_category) + 1.5), 0) + 0.001
        nums = np.floor(w / w.sum() * batch_num)
        nums[np.random.randint(num_category)] += (batch_num - nums.sum())
        X = []
        y = []
        for i in range(num_category):
            X.append(xList[i][np.random.randint(0,xList[i].shape[0], int(nums[i]))])
            y.append(np.tile(yList[i], (int(nums[i]), 1)))
        X = np.vstack(X)
        y = np.vstack(y)
        yield (X, y)