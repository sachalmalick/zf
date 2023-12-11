import constants as const
import numpy as np
import pickle

def get_call_id(example):
    for k in const.CALL_MAP:
        if(example.call_type in const.CALL_MAP[k]):
            return const.CALL_IDS[k]
        
def randomize_feature(x, i):
    minv = np.min(x[i], axis=0)
    maxv = np.max(x[i], axis=0)
    random_numbers = np.random.rand(x.shape[0])
    random_numbers = random_numbers * abs(maxv - minv)
    random_numbers = random_numbers + minv
    x[:,i] = random_numbers
    return x

def save_obj(obj, fn):
    with open(fn + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
    