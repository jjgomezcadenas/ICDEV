import numpy as np

from typing      import NamedTuple
from typing      import Tuple
from typing      import List
from   invisible_cities.evm  .ic_containers  import Measurement
from   invisible_cities.types.ic_types import minmax

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

class XYRanges(NamedTuple):
    X  : Tuple[float]
    Y  : Tuple[float]

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


class KrRanges(NamedTuple):
    E   : Tuple[float]
    Q   : Tuple[float]
    Z   : Tuple[float]
    XY  : Tuple[float]


class KrFit(NamedTuple):
    par  : np.array
    err  : np.array
    chi2 : float


class KrLTSlices(NamedTuple):
    Es    : np.array
    LT    : np.array
    chi2  : np.array
    valid : np.array

class KrLTLimits(NamedTuple):
    Es  : minmax
    LT  : minmax
    Eu  : minmax
    LTu : minmax


class KrMeanAndStd(NamedTuple):
    mu    : float
    std   : float
    mu_u  : float
    std_u : float


class KrMeansAndStds(NamedTuple):
    mu    : np.array
    std   : np.array
    mu_u  : np.array
    std_u : np.array
