import numpy as np
import glob

from   invisible_cities.io.dst_io  import load_dsts
from   invisible_cities.core.core_functions import loc_elem_1d


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def read_dsts(path_to_dsts):
    filenames = glob.glob(path_to_dsts+'/*')
    dstdf = load_dsts(filenames, group='DST', node='Events')
    nsdf  = load_dsts(filenames, group='Extra', node='nS12')
    return nsdf, dstdf


def bin_ratio(array, bins, xbin):
    return array[loc_elem_1d(bins, xbin)] / np.sum(array)


def bin_to_last_ratio(array, bins, xbin):
    return np.sum(array[loc_elem_1d(bins, xbin): -1]) / np.sum(array)
