import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from   invisible_cities.core.core_functions import weighted_mean_and_var
from   invisible_cities.core.core_functions import loc_elem_1d
from   invisible_cities.core.core_functions import in_range
from   invisible_cities.evm  .ic_containers  import Measurement
from   invisible_cities.io.dst_io  import load_dsts
from   invisible_cities.core.system_of_units_c import units
import invisible_cities.core.fit_functions as fitf

from icaro.core.fit_functions import quick_gauss_fit
from icaro.core.fit_functions import fit_slices_2d_expo
from icaro.core.fit_functions import expo_seed, gauss_seed
from icaro.core.fit_functions import to_relative
from icaro.core.fit_functions import conditional_labels

from invisible_cities.icaro. hst_functions import shift_to_bin_centers
from icaro.core.fit_functions import fit_profile_1d_expo

from typing      import NamedTuple
from typing      import Tuple
from typing      import List

from . kr_types import KrEvent
from . kr_types import KrRanges
from . kr_types import KrNBins
from . kr_types import KrBins
from . kr_types import KrFit
from . kr_types import XYRanges
from . kr_types import Ranges

with_titles  = True
labels = conditional_labels(with_titles)


def ns1(nsdf):
    hns1, bins  = histo_ns1(nsdf.ns1)
    print(' 0S1/tot  = {} 1S1/tot = {} 2S1/tot = {}'.format(bin_ratio(hns1, bins, 0),
                                                            bin_ratio(hns1, bins, 1),
                                                            bin_ratio(hns1, bins, 2)))


def read_dsts(path_to_dsts):
    filenames = glob.glob(path_to_dsts+'/*')
    dstdf = load_dsts(filenames, group='DST', node='Events')
    nsdf  = load_dsts(filenames, group='Extra', node='nS12')
    return nsdf, dstdf


def bin_ratio(array, bins, xbin):
    return array[loc_elem_1d(bins, xbin)] / np.sum(array)


def bin_to_last_ratio(array, bins, xbin):
    return np.sum(array[loc_elem_1d(bins, xbin): -1]) / np.sum(array)


def ns1_stats(nsdf):
    mu, var = weighted_mean_and_var(nsdf.ns1, np.ones(len(nsdf.ns1)))
    hist, bins = np.histogram(nsdf.ns1, bins = 20, range=(0,20))
    s1r = [bin_ratio(hist, bins, i) for i in range(0,4)]
    s1r.append(bin_to_last_ratio(hist, bins, 4))
    return mu, var, s1r


def ns2_stats(nsdf):
    mu, var = weighted_mean_and_var(nsdf.ns2, np.ones(len(nsdf.ns2)))
    hist, bins = np.histogram(nsdf.ns2, bins = 5, range=(0,5))
    s1r = [bin_ratio(hist, bins, i) for i in range(0,3)]
    s1r.append(bin_to_last_ratio(hist, bins, 3))
    return mu, var, s1r


def print_ns12_stats(mu, var, s1r):
    print('ns12: mean = {:5.2f} sigma = {:5.2f}'.format(mu, np.sqrt(var)))
    print('ns12 : fraction')
    print('\n'.join('{}: {:5.2f}'.format(*k) for k in enumerate(s1r)))


def ns1s(rnd, figsize=(6,6)):
    vals = rnd.values()
    labels = [x.label for x in vals]
    dsts   = [x.dst for x in vals]
    fig = plt.figure(figsize=figsize) # Creates a new figure

    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel('n S1',fontsize = 11)
    ax.set_ylabel('Frequency', fontsize = 11)
    hns1, bins, _ = ax.hist([df.ns1.values for df in dsts], bins = 20, range=(0,20),
            histtype='step',
            label=labels,
            linewidth=1.5)
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)


def get_stats(rnd, i, f, r=False):
    xr = list(range(i,f))
    if r:
        xr = list(reversed(xr))
    stats = [ns1_stats(rnd[i].ns) for i in xr]
    return stats, xr


def plot_srs(rnd, ri, rf, reverse=False):
    stats, xr =get_stats(rnd, ri, rf, reverse)
    srs =[stats[i][2] for i in range(len(stats))]

    for j, i in enumerate(xr):
        plt.plot(srs[j], label=rnd[i].label)
    plt.xlabel('ns1')
    plt.ylabel('fraction')
    plt.grid(True)
    plt.legend(fontsize= 10, loc='upper right')


def plot_mus(rnd, ri, rf, reverse=False):
    stats, xr =get_stats(rnd, ri, rf, reverse)
    mus = [stats[i][0] for i in range(len(stats))]

    plt.plot(xr, mus)
    plt.xlabel('run number')
    plt.ylabel('mean')
    plt.grid(True)
        #plt.legend(fontsize= 10, loc='upper right')


def print_stats(rn, rnd, mu, var, s1r):
    lbl = rnd[rn].label
    print(lbl)
    print('ns1: mean = {:5.2f} sigma = {:5.2f}'.format(mu, np.sqrt(var)))
    print('ns1 : fraction')
    print('\n'.join('{}: {:5.2f}'.format(*k) for k in enumerate(s1r)))



def h1d(x, bins=None, range=None, xlabel='Variable', ylabel='Frequency',
        title=None, legend = 'upper right', figsize=(6,6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    mu, var = weighted_mean_and_var(x, np.ones(len(x)))
    ax.set_xlabel(xlabel,fontsize = 11)
    ax.set_ylabel(ylabel, fontsize = 11)
    ax.hist(x,
            bins= bins,
            range=range,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc=legend)
    plt.grid(True)

    if title:
        plt.title(title)


def h2d(x, y, bins=None, xrange=None, yrange=None,
       xlabel='X', ylabel='Y', title=None, figsize=(8,8)):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    if xrange and yrange:
        plt.hist2d(x, y, bins=bins, range=(xrange,yrange))
    else:
        plt.hist2d(x, y, bins=bins)
    plt.colorbar()
    ax.set_xlabel(xlabel ,fontsize = 11) #xlabel
    ax.set_ylabel(ylabel, fontsize = 11)
    plt.grid(True)
    if title:
        plt.title(title)


def histo_ns12(ns1,
               xlabel='n S12', ylabel='arbitrary units',
               title = 'ns12', norm=1, fontsize = 11, figsize=(6,6)):
    fig = plt.figure(figsize=figsize) # Creates a new figure

    ax = fig.add_subplot(1, 1, 1)
    #mu, var = weighted_mean_and_var(ns1, np.ones(len(ns1)))
    ax.set_xlabel(xlabel,fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)
    ax.set_title(title, fontsize = 12)
    ax.hist(ns1,
            normed=norm,
            bins = 10,
            range=(0,10),
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label='# S1 candidates')
    #ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)
    #return hns1, bins


def plot_s1histos(s1df, bins=20, figsize=(12,12)):

    fig = plt.figure(figsize=figsize) # Creates a new figure

    ax = fig.add_subplot(3, 2, 1)
    mu, var = weighted_mean_and_var(s1df.es1.values, np.ones(len(s1df)))
    ax.set_xlabel('S1 energy (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)#ylabel
    ax.hist(s1df.es1,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 2)
    mu, var = weighted_mean_and_var(s1df.ws1, np.ones(len(s1df)))
    ax.set_xlabel(r'S1 width ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(s1df.ws1,
            range=(0,500),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 3)
    mu, var = weighted_mean_and_var(s1df.hs1, np.ones(len(s1df)))
    ax.set_xlabel(r'S1 height (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(s1df.hs1,
            range=(0,10),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 4)
    ok         = s1df.hs1.values > 0
    hr        = np.zeros(len(s1df.hs1.values))
    np.divide(s1df.hs1.values, s1df.es1.values, out=hr, where=ok)
    mu, var = weighted_mean_and_var(hr, np.ones(len(s1df)))
    ax.set_xlabel(r'height / energy ',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(hr,
            range=(0,1),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 5)
    mu, var = weighted_mean_and_var(s1df.ts1, np.ones(len(s1df)))
    ax.set_xlabel(r'S1 time ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(s1df.ts1,
            range=(0,600),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    #ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 6)
    plt.hist2d(s1df.ts1, s1df.es1, bins=10, range=((0,600),(0,30)))
    plt.colorbar()
    ax.set_xlabel(r'S1 time ($\mu$s) ',fontsize = 11) #xlabel
    ax.set_ylabel('S1 height (pes)', fontsize = 11)
    plt.grid(True)

    plt.tight_layout()

def plot_s1histos_short(s1df, bins=20, figsize=(12,12)):

    fig = plt.figure(figsize=figsize) # Creates a new figure

    ax = fig.add_subplot(4, 2, 1)
    mu, var = weighted_mean_and_var(s1df.es1.values, np.ones(len(s1df)))
    ax.set_xlabel('S1 energy (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('number of events', fontsize = 11)#ylabel
    ax.hist(s1df.es1,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(4, 2, 2)
    mu, var = weighted_mean_and_var(s1df.ws1, np.ones(len(s1df)))
    ax.set_xlabel(r'S1 width ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('number of events', fontsize = 11)
    ax.hist(s1df.ws1,
            range=(0,500),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(4, 2, 3)
    mu, var = weighted_mean_and_var(s1df.hs1, np.ones(len(s1df)))
    ax.set_xlabel(r'S1 height (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('number of events', fontsize = 11)
    ax.hist(s1df.hs1,
            range=(0,10),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(4, 2, 4)
    mu, var = weighted_mean_and_var(s1df.ts1, np.ones(len(s1df)))
    ax.set_xlabel(r'S1 time ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('number of events', fontsize = 11)
    ax.hist(s1df.ts1,
            range=(0,600),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    #ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    plt.tight_layout()


def plot_s1histos_multi(s1df, s2df, s3df, s4df, bins=20, figsize=(12,12)):

    fig = plt.figure(figsize=figsize) # Creates a new figure

    ax = fig.add_subplot(4, 2, 1)
    #mu, var = weighted_mean_and_var(s1df.es1.values, np.ones(len(s1df)))
    ax.set_xlabel('S1 energy (pes)',fontsize = 11)
    ax.set_ylabel('arbitrary units', fontsize = 11)
    ax.hist(s1df.es1,
            normed=1,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='blue',
            linestyle='--',
            linewidth=1.5,
            label='inclusive')
    ax.hist(s2df.es1,
            normed=1,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linestyle='-',
            linewidth=1.5,
            label='1 S1')
    ax.hist(s3df.es1,
            normed=1,
            range=(0,30),
            bins=bins,
            histtype='step',
            linestyle='-.',
            edgecolor='red',
            linewidth=1.5,
            label='2 S1')
    ax.hist(s4df.es1,
            normed=1,
            range=(0,30),
            bins=bins,
            histtype='step',
            linestyle=':',
            edgecolor='green',
            linewidth=1.5,
            label='> 2 S1')
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(4, 2, 2)
    #mu, var = weighted_mean_and_var(s1df.ws1, np.ones(len(s1df)))
    ax.set_xlabel(r'S1 width ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('arbitrary units', fontsize = 11)
    ax.hist(s1df.ws1,
            normed=1,
            range=(0,500),
            bins=bins,
            histtype='step',
            edgecolor='blue',
            linestyle='--',
            linewidth=1.5,
            label='inclusive')
    ax.legend(fontsize= 10, loc='upper right')
    ax.hist(s2df.ws1,
            normed=1,
            range=(0,500),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linestyle='-',
            linewidth=1.5,
            label='1 S1')
    ax.legend(fontsize= 10, loc='upper right')
    ax.hist(s3df.ws1,
            normed=1,
            range=(0,500),
            bins=bins,
            histtype='step',
            edgecolor='red',
            linestyle='-.',
            linewidth=1.5,
            label='2 S1')
    ax.legend(fontsize= 10, loc='upper right')
    ax.hist(s4df.ws1,
            normed=1,
            range=(0,500),
            bins=bins,
            histtype='step',
            edgecolor='green',
            linestyle=':',
            linewidth=1.5,
            label=' > 2 S1')
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(4, 2, 3)
    #mu, var = weighted_mean_and_var(s1df.hs1, np.ones(len(s1df)))
    ax.set_xlabel(r'S1 height (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('arbitrary units', fontsize = 11)
    ax.hist(s1df.hs1,
            normed=1,
            range=(0,10),
            bins=bins,
            histtype='step',
            edgecolor='blue',
            linestyle='--',
            linewidth=1.5,
            label=r'inclusive')
    ax.legend(fontsize= 10, loc='upper right')
    ax.hist(s2df.hs1,
            normed=1,
            range=(0,10),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linestyle='-',
            linewidth=1.5,
            label=r'1 S1')
    ax.legend(fontsize= 10, loc='upper right')
    ax.hist(s3df.hs1,
            normed=1,
            range=(0,10),
            bins=bins,
            histtype='step',
            edgecolor='red',
            linestyle='-.',
            linewidth=1.5,
            label=r'2 S1')
    ax.legend(fontsize= 10, loc='upper right')
    ax.hist(s4df.hs1,
            normed=1,
            range=(0,10),
            bins=bins,
            histtype='step',
            edgecolor='green',
            linestyle=':',
            linewidth=1.5,
            label=r'> 2 S1')
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(4, 2, 4)
    #mu, var = weighted_mean_and_var(s1df.ts1, np.ones(len(s1df)))
    ax.set_xlabel(r'S1 time ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('arbitrary units', fontsize = 11)
    ax.hist(s1df.ts1,
            normed = 1,
            range=(0,600),
            bins=bins,
            histtype='step',
            edgecolor='blue',
            linestyle='--',
            linewidth=1.5,
            label=r'inclusive')
    #ax.legend(fontsize= 10, loc='upper right')
    ax.hist(s1df.ts1,
            normed = 1,
            range=(0,600),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linestyle='-',
            linewidth=1.5,
            label=r'1 S1')
    #ax.legend(fontsize= 10, loc='upper right')
    ax.hist(s1df.ts1,
            normed = 1,
            range=(0,600),
            bins=bins,
            histtype='step',
            edgecolor='red',
            linestyle='-.',
            linewidth=1.5,
            label=r'2 S1')
    #ax.legend(fontsize= 10, loc='upper right')
    ax.hist(s1df.ts1,
            normed = 1,
            range=(0,600),
            bins=bins,
            histtype='step',
            edgecolor='green',
            linestyle=':',
            linewidth=1.5,
            label=r'> 2 S1')
    #ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    plt.tight_layout()

def plot_s2histos(df, bins=20, emin=3000, emax=15000, figsize=(12,12)):

    fig = plt.figure(figsize=figsize) # Creates a new figure

    ax = fig.add_subplot(3, 2, 1)
    mu, var = weighted_mean_and_var(df.es2.values, np.ones(len(df)))
    ax.set_xlabel('S2 energy (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)#ylabel
    ax.hist(df.es2,
            range=(emin, emax),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 2)
    mu, var = weighted_mean_and_var(df.ws2/units.mus, np.ones(len(df)))
    ax.set_xlabel(r'S2 width ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(df.ws2/units.mus,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 3)
    mu, var = weighted_mean_and_var(df.qs2, np.ones(len(df)))
    ax.set_xlabel(r'Q (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(df.qs2,
            range=(0,1000),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 4)
    mu, var = weighted_mean_and_var(df.nsi, np.ones(len(df)))
    ax.set_xlabel(r'number SiPM',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(df.nsi,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 5)
    mu, var = weighted_mean_and_var(df.xs2, np.ones(len(df)))
    ax.set_xlabel(r' X (mm)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(df.xs2,
            range=(-200,200),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 6)
    mu, var = weighted_mean_and_var(df.ys2, np.ones(len(df)))
    ax.set_xlabel(r' Y (mm)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(df.ys2,
            range=(-200,200),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, np.sqrt(var)))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    plt.tight_layout()


def xy_event_map(kre : KrEvent, kB: KrBins) -> np.array :
    """Plots XY map and returns number of events per bin"""

    nevt, *_ = plt.hist2d(kre.X, kre.Y, (kB.XY, kB.XY))
    XYpitch   = np.diff(kB.XY)[0]
    plt.colorbar().set_label("Number of events")
    labels("X (mm)", "Y (mm)", f"Event distribution for {XYpitch:.1f} mm pitch")
    return nevt

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
                slope [i, j] = abs(f.values[1])
                slopeu[i, j] = f.errors[1]
                chi2  [i, j] = f.chi2
                valid [i, j] = True
            except:
                pass
    return Measurement(const, constu), Measurement(slope, slopeu), chi2, valid
