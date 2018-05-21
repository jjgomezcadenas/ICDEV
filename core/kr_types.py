import numpy as np

from typing      import NamedTuple
from typing      import Tuple
from typing      import List

class KrEvent(NamedTuple):
    E : np.array
    Q : np.array
    X : np.array
    Y : np.array
    Z : np.array

class KrRanges(NamedTuple):
    E  : Tuple[float]
    Q  : Tuple[float]
    Z  : Tuple[float]
    XY : Tuple[float]

class Ranges(NamedTuple):
    lower  : Tuple[float]
    upper  : Tuple[float]

class KrNBins(NamedTuple):
    E  : int
    Q  : int
    Z  : int
    XY : int

class KrBins(NamedTuple):
    E   : np.array
    Q   : np.array
    Z   : np.array
    XY  : np.array
    cXY : np.array

class KrFit(NamedTuple):
    par  : np.array
    err  : np.array
    chi2 : np.array
    valid: np.array
