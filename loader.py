import re
import pathlib
import pandas as pd
import constants as const
import librosa as libr
import numpy as np

class ZFinchDataset():
    def __init__(self, adult_paths):
        self.adult_paths = list(pathlib.Path(adult_paths).iterdir())
        self.adult_examples = [to_example(path) for path in self.adult_paths]
        self.examples = self.adult_examples

class ZFinchExample:
    def __init__(self, path, name, date, call_type, rendition_num):
        self.path = path
        self.name = name
        self.date = date
        self.call_type = call_type
        self.rendition_num = rendition_num

    def __str__(self):
        return "name: {}, date: {}, call_type: {}, rendition_num: {}".format(
                            self.name, self.date, self.call_type, self.rendition_num)

def to_example(path):
    name, date, call_type, rendition_num = re.match(const.FILE_NAME_PATTERN, path.name).groups()
    return ZFinchExample(path, name, date, call_type, rendition_num)

def load_dataset():
    return ZFinchDataset(const.ADULT_RECORDINGS_PATH)


