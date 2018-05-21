import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   invisible_cities.core.core_functions import weighted_mean_and_var
from   invisible_cities.core.core_functions import loc_elem_1d
from   invisible_cities.core.core_functions import in_range
from   invisible_cities.core.system_of_units_c import units
import invisible_cities.core.fit_functions as fitf

from icaro.core.fit_functions import quick_gauss_fit
from icaro.core.fit_functions import fit_slices_2d_expo
from icaro.core.fit_functions import expo_seed, gauss_seed
from icaro.core.fit_functions import to_relative
from icaro.core.fit_functions import conditional_labels
from icaro.core.fit_functions import fit_profile_1d_expo

from invisible_cities.icaro. hst_functions import shift_to_bin_centers
from   invisible_cities.evm  .ic_containers  import Measurement

from . kr_types import KrEvent
from . kr_types import KrRanges
from . kr_types import KrNBins
from . kr_types import KrBins
from . kr_types import KrRanges
from . kr_types import Ranges
from . kr_types import KrFit


def profile_and_fit(X, Y, xrange, yrange, nbins, fitpar, label):
    fitOpt  = "r"
    xe = (xrange[1] - xrange[0])/nbins

    x, y, sy = fitf.profileX(X, Y, nbins=nbins,
                             xrange=xrange, yrange=yrange, drop_nan=True)
    sel  = in_range(x, xrange[0], xrange[1])
    x, y, sy = x[sel], y[sel], sy[sel]
    f = fitf.fit(fitf.expo, x, y, fitpar, sigma=sy)

    plt.errorbar(x=x, xerr=xe, y=y, yerr=sy,
                 linestyle='none', marker='.')
    plt.plot(x, f.fn(x), fitOpt)
    #set_plot_labels(xlabel=label[0], ylabel=label[1], grid=True)
    return f, x, y, sy


def print_fit(f):
    for i, val in enumerate(f.values):
        print('fit par[{}] = {} error = {}'.format(i, val, f.errors[i]))


def chi2(F, X, Y, SY):
    fitx = F.fn(X)
    n = len(F.values)
    #print('degrees of freedom = {}'.format(n))
    chi2t = 0
    for i, x in enumerate(X):
        chi2 = abs(Y[i] - fitx[i])/SY[i]
        chi2t += chi2
        #print('x = {} f(x) = {} y = {} ey = {} chi2 = {}'.format(
               #x, fitx[i], Y[i], SY[i], chi2 ))
    return chi2t/(len(X)-n)


def fit_lifetime_from_profile(kre : KrEvent,
                              kR  : KrRanges,
                              kNB : KrNBins,
                              kB  : KrBins,
                              kL  : KrRanges,
                              title="Lifetime Fit")->KrFit:

    sel  = in_range(kre.X, *kL.XY) & in_range(kre.Y, *kL.XY)
    z, e = kre.Z[sel], kre.E[sel]

    frame_data = plt.gcf().add_axes((.1, .35,
                                 .8, .6))
    plt.hist2d(z, e, (kB.Z, kB.E))

    x, y, yu = fitf.profileX(z, e, kNB.Z, kR.Z, kR.E)
    plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)

    seed = expo_seed(x, y)
    f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)

    plt.plot(x, f.fn(x), "r-", lw=4)

    frame_data.set_xticklabels([])
    labels("", "Energy (pes)", title)
    lims = plt.xlim()

    frame_res = plt.gcf().add_axes((.1, .1,
                                .8, .2))
    plt.errorbar(x, (f.fn(x) - y) / yu, 1, np.diff(x)[0] / 2, fmt="p", c="k")
    plt.plot(lims, (0, 0), "g--")
    plt.xlim(*lims)
    plt.ylim(-5, +5)
    labels("Drift time (µs)", "Standarized residual")
    print_fit(f)
    print('chi2 = {}'.format(chi2(f, x, y, yu)))

    return KrFit(par  = np.array(f.values[1]),
                 err  = np.array(f.errors[1]),
                 chi2 = np.array(chi2(f, x, y, yu)),
                 valid= np.ones(1))


def fit_s2_energy_in_z_bins_within_XY_limits(kre : KrEvent,
                                             kL  : KrRanges,
                                             kNB : KrNBins,
                                             kB  : KrBins,
                                             eR  : Ranges,
                                             figsize=(12,12)) ->KrFit:
    fig = plt.figure(figsize=figsize) # Creates a new figure
    EMU  = []
    ES   = []
    CHI2 = []

    for j, i in enumerate(range(kNB.Z)):
        ax = fig.add_subplot(5, 2, i+1)
        ax.set_xlabel('S2 energy (pes)',fontsize = 11) #xlabel
        ax.set_ylabel('Number of events', fontsize = 11)#ylabel

        zlim=kB.Z[i], kB.Z[i+1]

        sel  = in_range(kre.X, *kL.XY) & in_range(kre.Y, *kL.XY) & in_range(kre.Z, *zlim)
        e =  kre.E[sel]

        print(f'bin : {j}, energy range for fit: lower = {eR.lower[j]}, upper = {eR.upper[j]}')
        sel = in_range(e, eR.lower[j], eR.upper[j])
        er = e[sel]

        y, b ,_ = ax.hist(er,
                          bins=kB.E,
                          histtype='step',
                          edgecolor='black',
                          linewidth=1.5)

        x = shift_to_bin_centers(b)

        df = pd.DataFrame(dict(y = y, x = x))
        df = df[df.y >0]
        fit_range = (df.x.values[0], df.x.values[-1])
        x = df.x.values
        y = df.y.values
        yu = np.sqrt(y)

        seed = gauss_seed(x, y)
        f  = fitf.fit(fitf.gauss, x, y, seed, fit_range=fit_range, sigma=yu)
        plt.plot(x, f.fn(x), "r-", lw=4)

        EMU.append(f.values[1])
        ES.append(f.errors[1])
        CHI2.append(chi2(f, x, y, yu))

        plt.grid(True)
    plt.tight_layout()

    return KrFit(par  = np.array(EMU),
                 err  = np.array(ES),
                 chi2 = np.array(CHI2),
                 valid= np.ones(len(EMU)))


def fit_lifetime(kB, kf, title="Lifetime Fit"):
    x =  shift_to_bin_centers(kB.Z)
    y =  kf.par
    yu = kf.err
    plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)
    seed = expo_seed(x, y)
    f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
    plt.plot(x, f.fn(x), "r-", lw=4)
    print_fit(f)
    print('chi2 = {}'.format(chi2(f, x, y, yu)))


def fit_lifetimes_from_profile(kre : KrEvent,
                               kR  : KrRanges,
                               kNB : KrNBins,
                               kB  : KrBins,
                               kL  : KrRanges,
                               min_entries = 1e2):

    nbins   = kNB.XY, kNB.XY
    const   = np.zeros(nbins)
    slope   = np.zeros(nbins)
    constu  = np.zeros(nbins)
    slopeu  = np.zeros(nbins)
    chi2    = np.zeros(nbins)
    valid   = np.zeros(nbins, dtype=bool)

    zrange = kL.Z

    #print(zrange)
    #print(kNB.Z)

    for i in range(kNB.XY):
        #print(f' i = {i}')
        xr = kB.XY[i:i + 2]
        #print(f'range x = {xr}')

        sel_x = in_range(kre.X, *kB.XY[i:i + 2])
        xxx = np.count_nonzero([x for x in sel_x if x == True])
        #print(xxx)
        for j in range(kNB.XY):
            #print(f' j = {j}')
            yr = kB.XY[j:j + 2]
            #print(f'range y = {yr}')

            sel_y = in_range(kre.Y, *kB.XY[j:j + 2])
            xxx = np.count_nonzero([x for x in sel_y if x == True])
            #print(xxx)

            sel   = sel_x & sel_y
            mine = np.count_nonzero(sel)
            #print(f'min entries = {mine}')
            if  mine < min_entries: continue
            #print('trying to fit')
            try:
                f = fit_profile_1d_expo(kre.Z[sel], kre.E[sel], kNB.Z, xrange=zrange)
                #print(f' f = {f}')
                const [i, j] = f.values[0]
                constu[i, j] = f.errors[0]
                slope [i, j] = abs(f.values[1])
                slopeu[i, j] = f.errors[1]
                chi2  [i, j] = f.chi2
                valid [i, j] = True
            except:
                print('error')
                pass

    return  Measurement(const, constu), Measurement(slope, slopeu), chi2, valid


def fit_slices_2d_expo(xdata, ydata, zdata, tdata,
                       xbins, ybins, nbins_z, zrange=None,
                       min_entries = 1e2):
    """
    Slice the data in x and y, make the profile in z of t,
    fit it to a exponential and return the relevant values.

    Parameters
    ----------
    xdata, ydata, zdata, tdata: array_likes
        Values of each coordinate.
    xbins, ybins: array_like
        The bins in the x coordinate.
    nbins_z: int
        The number of bins in the z coordinate for the profile.
    zrange: length-2 tuple (optional)
        Fix the range in z. Default is computed from min and max
        of the input data.
    min_entries: int (optional)
        Minimum amount of entries to perform the fit.

    Returns
    -------
    const: Measurement(np.ndarray, np.ndarray)
        Values of const with errors.
    slope: Measurement(np.ndarray, np.ndarray)
        Values of slope with errors.
    chi2: np.ndarray
        Chi2 from each fit.
    valid: boolean np.ndarray
        Where the fit has been succesfull.
    """
    nbins_x = np.size (xbins) - 1
    nbins_y = np.size (ybins) - 1
    nbins   = nbins_x, nbins_y
    const   = np.zeros(nbins)
    slope   = np.zeros(nbins)
    constu  = np.zeros(nbins)
    slopeu  = np.zeros(nbins)
    chi2    = np.zeros(nbins)
    valid   = np.zeros(nbins, dtype=bool)

    if zrange is None:
        zrange = np.min(zdata), np.max(zdata)
    for i in range(nbins_x):
        sel_x = in_range(xdata, *xbins[i:i + 2])
        for j in range(nbins_y):
            sel_y = in_range(ydata, *ybins[j:j + 2])
            sel   = sel_x & sel_y
            if np.count_nonzero(sel) < min_entries: continue

            try:
                f = fit_profile_1d_expo(zdata[sel], tdata[sel], nbins_z, xrange=zrange)
                const [i, j] = f.values[0]
                constu[i, j] = f.errors[0]
                slope [i, j] = -f.values[1]
                slopeu[i, j] = f.errors[1]
                chi2  [i, j] = f.chi2
                valid [i, j] = True
            except:
                pass
    return Measurement(const, constu), Measurement(slope, slopeu), chi2, valid
