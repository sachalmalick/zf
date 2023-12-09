import numpy as np

import librosa as libr
import tensorflow as tf
import pathlib
import sys

def load_data_trf(dir):
    paths = list(pathlib.Path(dir).iterdir())
    ds = [libr.load(path)[0] for path in paths]
    x = np.array(ds)
    return np.reshape(x, (x.shape[0], x.shape[1], 1))