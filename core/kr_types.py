import numpy as np

from typing      import NamedTuple
from typing      import Tuple
from typing      import List
from   invisible_cities.evm  .ic_containers  import Measurement
from   invisible_cities.types.ic_types import minmax
from invisible_cities.icaro. hst_functions import shift_to_bin_centers
import datetime

def kr_event(dst):

    dst_time = dst.sort_values('event')
    return KrEvent(X  = dst.X.values,
                   Y  = dst.Y.values,
                   Z  = dst.Z.values,
                   T  = dst_time.time.values,
                   E  = dst.S2e.values,
                   S1 = dst.S1e.values,
                   Q  = dst.S2q.values)

def kr_times_ranges_and_bins(dst,
                        Zrange  = ( 100,  550),
                        XYrange = (-220,  220),
                        Erange  = ( 2e3, 15e3),
                        S1range = (   0,   50),
                        Qrange  = ( 100, 1500),
                        Znbins        =   10,
                        XYnbins       =   30,
                        Enbins        =   50,
                        S1nbins       =   10,
                        Qnbins        =   25,
                        nStimeprofile = 3600
                       ):


        Zbins      = np.linspace(* Zrange,  Znbins + 1)
        Ebins      = np.linspace(* Erange,  Enbins + 1)
        S1bins     = np.linspace(* S1range,  S1nbins + 1)
        Qbins      = np.linspace(* Qrange,  Qnbins + 1)
        XYbins     = np.linspace(*XYrange, XYnbins + 1)
        XYcenters  = shift_to_bin_centers(XYbins)
        XYpitch    = np.diff(XYbins)[0]

        dst_time = dst.sort_values('event')
        T       = dst_time.time.values
        tstart  = T[0]
        tfinal  = T[-1]
        Trange  = (datetime.datetime.fromtimestamp(tstart),
                  datetime.datetime.fromtimestamp(tfinal))

        ntimebins  = int( np.floor( ( tfinal - tstart) / nStimeprofile) )
        Tnbins     = np.max([ntimebins, 1])
        Tbins      = np.linspace( tstart, tfinal, ntimebins+1)

        krNBins  = KrNBins(E = Enbins, S1=S1nbins, Q = Qnbins, Z = Znbins,
                           XY = XYnbins, T = Tnbins)
        krRanges = KrRanges(E = Erange, S1=S1range, Q = Qrange, Z = Zrange,
                           XY = XYrange, T = Trange)
        krBins   = KrBins(E = Ebins, S1=S1bins, Q = Qbins, Z = Zbins,
                          XY = XYbins, cXY = XYcenters, T = Tbins)

        times      = [np.mean([Tbins[t],Tbins[t+1]]) for t in range(Tnbins)]
        TL         = [(Tbins[t],Tbins[t+1]) for t in range(Tnbins)]
        timeStamps = list(map(datetime.datetime.fromtimestamp, times))
        krTimes    = KrTimes(times = times, timeStamps = timeStamps, TL = TL)

        return krTimes, krRanges, krNBins, krBins

class Ndst(NamedTuple):
    full  : int
    fid   : int
    core  : int
    hcore : int


class NevtDst(NamedTuple):
    full  : np.array
    fid   : np.array
    core  : np.array
    hcore : np.array


class EffDst(NamedTuple):
    full  : float
    fid   : float
    core  : float
    hcore : float


class KrEvent(NamedTuple):
    E  : np.array
    S1 : np.array
    Q  : np.array
    X  : np.array
    Y  : np.array
    Z  : np.array
    T  : np.array

class DstEvent(NamedTuple):
    full  : KrEvent
    fid   : KrEvent
    core  : KrEvent
    hcore : KrEvent


class KrRanges(NamedTuple):
    E  : Tuple[float]
    S1 : Tuple[float]
    Q  : Tuple[float]
    Z  : Tuple[float]
    T  : Tuple[float]
    XY : Tuple[float]


class Ranges(NamedTuple):
    lower  : Tuple[float]
    upper  : Tuple[float]


class XYRanges(NamedTuple):
    X  : Tuple[float]
    Y  : Tuple[float]


class KrNBins(NamedTuple):
    E  : int
    S1 : int
    Q  : int
    Z  : int
    T  : int
    XY : int


class KrBins(NamedTuple):
    E   : np.array
    S1  : np.array
    Q   : np.array
    Z   : np.array
    T   : np.array
    XY  : np.array
    cXY : np.array


class  KrTimes(NamedTuple):
    times      : List[float]
    timeStamps : List[float]
    TL         : List[Tuple[float]]


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
