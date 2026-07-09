import tensorflow as tf
print(tf.__version__)

import keras
print(keras.__version__)
from keras import layers
from keras.models import Sequential
from keras import utils
from keras.utils import to_categorical

import torch
print(torch.__version__)

# import pybaseball as pb
# from pybaseball import statcast
# print(pb.__version__)
# from pybaseball import cache

import pandas as pd
print(pd.__version__)

import numpy as np
print(np.__version__)

import os

import math

import sys

import matplotlib as mpl
print(mpl.__version__)


# data2 = statcast(start_dt = "2020-01-01", end_dt = "2026-01-01")
# cache.enable()
# data2.to_parquet("SC_PQ_Test.parquet", index=False)