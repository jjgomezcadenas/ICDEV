import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   invisible_cities.core.core_functions import weighted_mean_and_var
from   invisible_cities.core.core_functions import weighted_mean_and_std
from   invisible_cities.core.core_functions import loc_elem_1d
from   invisible_cities.core.core_functions import in_range
from   invisible_cities.core.system_of_units_c import units

from invisible_cities.icaro. hst_functions import shift_to_bin_centers
from . kr_types           import KrEvent
from . kr_types           import KrBins
from . kr_core_functions  import find_nearest


def h1d(x, bins=None, range=None, weights=None, xlabel='Variable', ylabel='Frequency',
        title=None, legend = 'upper right', figsize=(6,6)):

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    x1 = loc_elem_1d(x, find_nearest(x,range[0]))
    x2 = loc_elem_1d(x, find_nearest(x,range[1]))
    xmin = min(x1, x2)
    xmax = max(x1, x2)

    mu, std = weighted_mean_and_std(x[xmin:xmax], np.ones(len(x[xmin:xmax])))
    ax.set_xlabel(xlabel,fontsize = 11)
    ax.set_ylabel(ylabel, fontsize = 11)
    ax.hist(x,
            bins= bins,
            range=range,
            weights=weights,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, std))
    ax.legend(fontsize= 10, loc=legend)
    plt.grid(True)

    if title:
        plt.title(title)

    return mu, std


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


def xy_event_map(kre : KrEvent, kB: KrBins) -> np.array :
    """Plots XY map and returns number of events per bin"""

    nevt, *_ = plt.hist2d(kre.X, kre.Y, (kB.XY, kB.XY))
    XYpitch   = np.diff(kB.XY)[0]
    plt.colorbar().set_label("Number of events")
    labels("X (mm)", "Y (mm)", f"Event distribution for {XYpitch:.1f} mm pitch")
    return nevt


def h1d_4(h1ds,
          bins,
          ranges,
          xlabels,
          ylabels,
          titles  =None,
          legends = ('best','best','best','best'),
          figsize =(10,10)):

    fig = plt.figure(figsize=figsize)

    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1)
        x = h1ds[i]
        r = ranges[i]

        x1 = loc_elem_1d(x, find_nearest(x,r[0]))
        x2 = loc_elem_1d(x, find_nearest(x,r[1]))
        xmin = min(x1, x2)
        xmax = max(x1, x2)
        x2 = x[xmin:xmax]
        o  = np.ones(len(x2))
        mu, std = weighted_mean_and_std(x2, o)

        ax.set_xlabel(xlabels[i],fontsize = 11)
        ax.set_ylabel(ylabels[i], fontsize = 11)
        ax.hist(x,
                bins= bins[i],
                range=r,
                histtype='step',
                edgecolor='black',
                linewidth=1.5,
                label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, std))
        ax.legend(fontsize= 10, loc=legends[i])
        plt.grid(True)
        if titles:
            plt.title(titles[i])

    plt.tight_layout()
