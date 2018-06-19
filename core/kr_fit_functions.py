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
from . kr_types import XYRanges
from . kr_types import Ranges
from . kr_types import KrFit
from . kr_types import KrLTSlices

labels = conditional_labels(True)

def select_in_XYRange(kre : KrEvent, xyr : XYRanges)->KrEvent:
    """ Selects a KrEvent in  a range of XY values"""
    xr = xyr.X
    yr = xyr.Y
    sel  = in_range(kre.X, *xr) & in_range(kre.Y, *yr)

    return KrEvent(X = kre.X[sel],
                   Y = kre.Y[sel],
                   Z = kre.Z[sel],
                   E = kre.E[sel],
                   Q = kre.Q[sel])


def lifetime_in_XYRange(kre : KrEvent,
                        krnb: KrNBins,
                        krb : KrBins,
                        krr : KrRanges,
                        xyr : XYRanges)->KrFit:
    """ Plots lifetime fitted to a range of XY values"""

    # select data in region defined by xyr
    kre_xy = select_in_XYRange(kre, xyr)
    z, e = kre_xy.Z, kre_xy.E

    # Specify the range and number of bins in Z
    Znbins = krnb.Z
    Zrange = krr.Z

    # create a figure and plot 2D histogram and profile
    frame_data = plt.gcf().add_axes((.1, .3, .8, .6))
    plt.hist2d(z, e, (krb.Z, krb.E))
    x, y, yu = fitf.profileX(z, e, Znbins, Zrange)
    plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)

    # Fit profile to an exponential
    seed = expo_seed(x, y)
    f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)

    # plot fitted value
    plt.plot(x, f.fn(x), "r-", lw=4)

    # labels and ticks
    frame_data.set_xticklabels([])
    labels("", "Energy (pes)", "Lifetime fit")

    # add a second frame

    lims = plt.xlim()
    frame_res = plt.gcf().add_axes((.1, .1, .8, .2))
    # Plot (y - f(x)) / sigma(y) as a function of x
    plt.errorbar(x, (f.fn(x) - y) / yu, 1, np.diff(x)[0] / 2,
                 fmt="p", c="k")
    plt.plot(lims, (0, 0), "g--")
    plt.xlim(*lims)
    plt.ylim(-5, +5)
    labels("Drift time (µs)", "Standarized residual")

    return KrFit(par  = np.array(f.values),
                 err  = np.array(f.errors),
                 chi2 = chi2(f, x, y, yu))


def lifetimes_in_XYRange(kre : KrEvent,
                        krnb: KrNBins,
                        krb : KrBins,
                        krr : KrRanges,
                        xyr : XYRanges,
                        XL = [(-125, -75), (-125, -75), (75, 125),(75, 125)],
                        YL = [(-125, -75), (75, 125), (75, 125),(-125, -75)],
                        figsize=(8,8))->KrFit:
    """ Plots lifetime fitted to a range of XY values"""


    # Specify the range and number of bins in Z
    Znbins = krnb.Z
    Zrange = krr.Z

    fig = plt.figure(figsize=figsize)

    # XL = [(-125, -75), (-125, -75), (75, 125),(75, 125)]
    # YL = [(-125, -75), (75, 125), (75, 125),(-125, -75)]

    for i, pair in enumerate(zip(XL,YL)):
        xlim = pair[0]
        ylim = pair[1]
        print(f'xlim = {xlim}, ylim ={ylim}')

        # select data in region defined by xyr
        xyr = XYRanges(X=xlim, Y=ylim )
        kre_xy = select_in_XYRange(kre, xyr)
        z, e = kre_xy.Z, kre_xy.E

        ax = fig.add_subplot(2, 2, i+1)
        x, y, yu = fitf.profileX(z, e, Znbins, Zrange)
        plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)

        # Fit profile to an exponential
        seed = expo_seed(x, y)
        f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)

        # plot fitted value
        plt.plot(x, f.fn(x), "r-", lw=4)


        labels("", "Energy (pes)", "Lifetime fit")


        kf = KrFit(par  = np.array(f.values),
                   err  = np.array(f.errors),
                   chi2 = chi2(f, x, y, yu))


        print_fit(kf)



def fit_slices_2d_expo(kre : KrEvent,
                       krnb: KrNBins,
                       krb : KrBins,
                       krr : KrRanges,
                       fit_var = "E",
                       min_entries = 1e2)->KrLTSlices:

    """
    Slice the data in x and y, make the profile in z of E,
    fit it to a exponential and return the relevant values.

    """

    xbins   = krb.XY
    ybins   = krb.XY
    nbins_x = np.size (xbins) - 1
    nbins_y = np.size (ybins) - 1
    nbins_z = krnb.Z
    nbins   = nbins_x, nbins_y
    const   = np.zeros(nbins)
    slope   = np.zeros(nbins)
    constu  = np.zeros(nbins)
    slopeu  = np.zeros(nbins)
    chi2    = np.zeros(nbins)
    valid   = np.zeros(nbins, dtype=bool)
    zrange = krr.Z

    for i in range(nbins_x):
        sel_x = in_range(kre.X, *xbins[i:i + 2])
        for j in range(nbins_y):
            sel_y = in_range(kre.Y, *ybins[j:j + 2])
            sel   = sel_x & sel_y
            if np.count_nonzero(sel) < min_entries:
                print(f'entries ={entries} not enough  to fit bin (i,j) =({i},{j})')
                valid [i, j] = False
                continue

            try:
                z = kre.Z[sel]
                t = kre.E[sel]
                if fit_var == "Q":
                    t = kre.Q[sel]

                f = fit_profile_1d_expo(z, t, nbins_z, xrange=zrange)
                re = np.abs(f.errors[1] / f.values[1])

                if re > 0.5:
                    print(f'Relative error to large, re ={re} for bin (i,j) =({i},{j})')
                    valid [i, j] = False

                const [i, j] = f.values[0]
                constu[i, j] = f.errors[0]
                slope [i, j] = -f.values[1]
                slopeu[i, j] = f.errors[1]
                chi2  [i, j] = f.chi2
                valid [i, j] = True
            except:
                pass
    return KrLTSlices(Ez0  = Measurement(const, constu),
                       LT   = Measurement(slope, slopeu),
                       chi2 = chi2,
                       valid = valid)


def fit_and_plot_slices_2d_expo(kre : KrEvent,
                                krnb: KrNBins,
                                krb : KrBins,
                                krr : KrRanges,
                                fit_var = "E",
                                min_entries = 1e2,
                                figsize=(12,12))->KrLTSlices:

    """
    Slice the data in x and y, make the profile in z of E,
    fit it to a exponential and return the relevant values.

    """

    xybins   = krb.XY
    nbins_xy = np.size (xybins) - 1
    nbins_z = krnb.Z
    nbins   = nbins_xy, nbins_xy
    const   = np.zeros(nbins)
    slope   = np.zeros(nbins)
    constu  = np.zeros(nbins)
    slopeu  = np.zeros(nbins)
    chi2    = np.zeros(nbins)
    valid   = np.zeros(nbins, dtype=bool)
    zrange = krr.Z

    fig = plt.figure(figsize=figsize) # Creates a new figure
    k=0
    index = 0
    for i in range(nbins_xy):
        sel_x = in_range(kre.X, *xybins[i:i + 2])
        for j in range(nbins_xy):
            index +=1
            #print(f' bin =({i},{j});  index = {index}')
            if k%25 ==0:
                k=0
                fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(5, 5, k+1)
            k+=1
            sel_y = in_range(kre.Y, *xybins[j:j + 2])
            sel   = sel_x & sel_y
            entries = np.count_nonzero(sel)
            if entries < min_entries:
                print(f'entries ={entries} not enough  to fit bin (i,j) =({i},{j})')
                valid [i, j] = False
                continue

            try:
                z = kre.Z[sel]
                t = kre.E[sel]
                if fit_var == "Q":
                    t = kre.Q[sel]

                x, y, yu = fitf.profileX(z, t, nbins_z, zrange)
                ax.errorbar(x, y, yu, np.diff(x)[0]/2,
                             fmt="kp", ms=7, lw=3)
                seed = expo_seed(x, y)
                f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
                plt.plot(x, f.fn(x), "r-", lw=4)
                plt.grid(True)
                re = np.abs(f.errors[1] / f.values[1])
                #print(f' E +- Eu = {f.values[0]} +- {f.errors[0]}')
                #print(f' LT +- LTu = {-f.values[1]} +- {f.errors[1]}')
                #print(f' LTu/LT = {re} chi2 = {f.chi2}')

                if re > 0.5:
                    print(f'Relative error to large, re ={re} for bin (i,j) =({i},{j})')
                    print(f' LT +- LTu = {-f.values[1]} +- {f.errors[1]}')
                    print(f' LTu/LT = {re} chi2 = {f.chi2}')
                    valid [i, j] = False

                const [i, j] = f.values[0]
                constu[i, j] = f.errors[0]
                slope [i, j] = -f.values[1]
                slopeu[i, j] = f.errors[1]
                chi2  [i, j] = f.chi2
                valid [i, j] = True
            except:
                print(f'fit failed for bin (i,j) =({i},{j})')
                pass
    plt.tight_layout()

    return KrLTSlices(Ez0  = Measurement(const, constu),
                       LT   = Measurement(slope, slopeu),
                       chi2 = chi2,
                       valid = valid)

def fit_lifetime_slices(kre : KrEvent,
                        krnb: KrNBins,
                        krb : KrBins,
                        krr : KrRanges,
                        fit_var = "E",
                        min_entries = 1e2)->KrLTSlices:

    """
    Slice the data in x and y, make the profile in z of E,
    fit it to a exponential and return the relevant values.

    """

    xybins   = krb.XY
    nbins_xy = np.size (xybins) - 1
    nbins_z = krnb.Z
    nbins   = nbins_xy, nbins_xy
    const   = np.zeros(nbins)
    slope   = np.zeros(nbins)
    constu  = np.zeros(nbins)
    slopeu  = np.zeros(nbins)
    chi2    = np.zeros(nbins)
    valid   = np.zeros(nbins, dtype=bool)
    zrange = krr.Z

    for i in range(nbins_xy):
        sel_x = in_range(kre.X, *xybins[i:i + 2])
        for j in range(nbins_xy):
            #print(f' bin =({i},{j});  index = {index}')
            sel_y = in_range(kre.Y, *xybins[j:j + 2])
            sel   = sel_x & sel_y
            entries = np.count_nonzero(sel)
            if entries < min_entries:
                #print(f'entries ={entries} not enough  to fit bin (i,j) =({i},{j})')
                valid [i, j] = False
                continue

            try:
                z = kre.Z[sel]
                t = kre.E[sel]
                if fit_var == "Q":
                    t = kre.Q[sel]

                x, y, yu = fitf.profileX(z, t, nbins_z, zrange)

                seed = expo_seed(x, y)
                f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
                re = np.abs(f.errors[1] / f.values[1])
                #print(f' E +- Eu = {f.values[0]} +- {f.errors[0]}')
                #print(f' LT +- LTu = {-f.values[1]} +- {f.errors[1]}')
                #print(f' LTu/LT = {re} chi2 = {f.chi2}')

                const [i, j] = f.values[0]
                constu[i, j] = f.errors[0]
                slope [i, j] = -f.values[1]
                slopeu[i, j] = f.errors[1]
                chi2  [i, j] = f.chi2
                valid [i, j] = True

                if re > 0.5:
                    # print(f'Relative error to large, re ={re} for bin (i,j) =({i},{j})')
                    # print(f' LT +- LTu = {-f.values[1]} +- {f.errors[1]}')
                    # print(f' LTu/LT = {re} chi2 = {f.chi2}')
                    valid [i, j] = False

            except:
                print(f'fit failed for bin (i,j) =({i},{j})')
                pass

    return KrLTSlices(Es  = Measurement(const, constu),
                       LT   = Measurement(slope, slopeu),
                       chi2 = chi2,
                       valid = valid)


def print_fit(krf: KrFit):
    print(f' E (z=0) = {krf.par[0]} +-{krf.err[0]} ')
    print(f' LT      = {krf.par[1]} +-{krf.err[1]} ')
    print(f' chi2    = {krf.chi2} ')


def print_krfit(f):
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



# def fit_s2_energy_in_z_bins_within_XY_limits(kre : KrEvent,
#                                              kL  : KrRanges,
#                                              kNB : KrNBins,
#                                              kB  : KrBins,
#                                              eR  : Ranges,
#                                              figsize=(12,12)) ->KrFit:
#     fig = plt.figure(figsize=figsize) # Creates a new figure
#     EMU  = []
#     ES   = []
#     CHI2 = []
#
#     for j, i in enumerate(range(kNB.Z)):
#         ax = fig.add_subplot(5, 2, i+1)
#         ax.set_xlabel('S2 energy (pes)',fontsize = 11) #xlabel
#         ax.set_ylabel('Number of events', fontsize = 11)#ylabel
#
#         zlim=kB.Z[i], kB.Z[i+1]
#
#         sel  = in_range(kre.X, *kL.XY) & in_range(kre.Y, *kL.XY) & in_range(kre.Z, *zlim)
#         e =  kre.E[sel]
#
#         print(f'bin : {j}, energy range for fit: lower = {eR.lower[j]}, upper = {eR.upper[j]}')
#         sel = in_range(e, eR.lower[j], eR.upper[j])
#         er = e[sel]
#
#         y, b ,_ = ax.hist(er,
#                           bins=kB.E,
#                           histtype='step',
#                           edgecolor='black',
#                           linewidth=1.5)
#
#         x = shift_to_bin_centers(b)
#
#         df = pd.DataFrame(dict(y = y, x = x))
#         df = df[df.y >0]
#         fit_range = (df.x.values[0], df.x.values[-1])
#         x = df.x.values
#         y = df.y.values
#         yu = np.sqrt(y)
#
#         seed = gauss_seed(x, y)
#         f  = fitf.fit(fitf.gauss, x, y, seed, fit_range=fit_range, sigma=yu)
#         plt.plot(x, f.fn(x), "r-", lw=4)
#
#         EMU.append(f.values[1])
#         ES.append(f.errors[1])
#         CHI2.append(chi2(f, x, y, yu))
#
#         plt.grid(True)
#     plt.tight_layout()
#
#     return KrFit(par  = np.array(EMU),
#                  err  = np.array(ES),
#                  chi2 = np.array(CHI2),
#                  valid= np.ones(len(EMU)))


# def fit_lifetime(kB, kf, title="Lifetime Fit"):
#     x =  shift_to_bin_centers(kB.Z)
#     y =  kf.par
#     yu = kf.err
#     plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)
#     seed = expo_seed(x, y)
#     f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
#     plt.plot(x, f.fn(x), "r-", lw=4)
#     print_fit(f)
#     print('chi2 = {}'.format(chi2(f, x, y, yu)))


# def fit_lifetimes_from_profile(kre : KrEvent,
#                                kR  : KrRanges,
#                                kNB : KrNBins,
#                                kB  : KrBins,
#                                kL  : KrRanges,
#                                min_entries = 1e2):
#
#     nbins   = kNB.XY, kNB.XY
#     const   = np.zeros(nbins)
#     slope   = np.zeros(nbins)
#     constu  = np.zeros(nbins)
#     slopeu  = np.zeros(nbins)
#     chi2    = np.zeros(nbins)
#     valid   = np.zeros(nbins, dtype=bool)
#
#     zrange = kL.Z
#
#     #print(zrange)
#     #print(kNB.Z)
#
#     for i in range(kNB.XY):
#         #print(f' i = {i}')
#         xr = kB.XY[i:i + 2]
#         #print(f'range x = {xr}')
#
#         sel_x = in_range(kre.X, *kB.XY[i:i + 2])
#         xxx = np.count_nonzero([x for x in sel_x if x == True])
#         #print(xxx)
#         for j in range(kNB.XY):
#             #print(f' j = {j}')
#             yr = kB.XY[j:j + 2]
#             #print(f'range y = {yr}')
#
#             sel_y = in_range(kre.Y, *kB.XY[j:j + 2])
#             xxx = np.count_nonzero([x for x in sel_y if x == True])
#             #print(xxx)
#
#             sel   = sel_x & sel_y
#             mine = np.count_nonzero(sel)
#             #print(f'min entries = {mine}')
#             if  mine < min_entries: continue
#             #print('trying to fit')
#             try:
#                 f = fit_profile_1d_expo(kre.Z[sel], kre.E[sel], kNB.Z, xrange=zrange)
#                 #print(f' f = {f}')
#                 const [i, j] = f.values[0]
#                 constu[i, j] = f.errors[0]
#                 slope [i, j] = abs(f.values[1])
#                 slopeu[i, j] = f.errors[1]
#                 chi2  [i, j] = f.chi2
#                 valid [i, j] = True
#             except:
#                 print('error')
#                 pass
#
#     return  Measurement(const, constu), Measurement(slope, slopeu), chi2, valid
#
