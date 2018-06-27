import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates  as md
from invisible_cities.icaro.mpl_functions import set_plot_labels
from   invisible_cities.core.system_of_units_c import units

from . kr_types import KrLTLimits
from . kr_types import NevtDst
from invisible_cities.icaro. hst_functions import display_matrix
from   invisible_cities.evm  .ic_containers  import Measurement
from invisible_cities.icaro. hst_functions import display_matrix
from icaro.core.fit_functions import conditional_labels

labels = conditional_labels(True)


def figsize(type="small"):
    if type == "S":
        plt.rcParams["figure.figsize"]  = 8, 6
    elif type == "s":
         plt.rcParams["figure.figsize"] = 6, 4
    elif type == "l":
        plt.rcParams["figure.figsize"] = 10, 8
    else:
        plt.rcParams["figure.figsize"] = 12, 10

def plot_xy_density(kdst, krBins, XYpitch, figsize=(14,10)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt_full, *_ = plt.hist2d(kdst.full.X, kdst.full.Y, (krBins.XY, krBins.XY))
    plt.colorbar().set_label("Number of events")
    labels("X (mm)", "Y (mm)", f"full distribution for {XYpitch:.1f} mm pitch")

    fig.add_subplot(2, 2, 2)
    nevt_fid, *_ = plt.hist2d(kdst.fid.X, kdst.fid.Y, (krBins.XY, krBins.XY))
    plt.colorbar().set_label("Number of events")
    labels("X (mm)", "Y (mm)", f"fid distribution for {XYpitch:.1f} mm pitch")

    fig.add_subplot(2, 2, 3)
    nevt_core, *_ = plt.hist2d(kdst.core.X, kdst.core.Y, (krBins.XY, krBins.XY))
    plt.colorbar().set_label("Number of events")
    labels("X (mm)", "Y (mm)", f"core distribution for {XYpitch:.1f} mm pitch")

    fig.add_subplot(2, 2, 4)
    nevt_hcore, *_ = plt.hist2d(kdst.hcore.X, kdst.hcore.Y, (krBins.XY, krBins.XY))
    plt.colorbar().set_label("Number of events")
    labels("X (mm)", "Y (mm)", f"hard core distribution for {XYpitch:.1f} mm pitch")
    plt.tight_layout()
    return NevtDst(full  = nevt_full,
                   fid   = nevt_fid,
                   core  = nevt_core,
                   hcore = nevt_hcore)

def plot_s2_vs_z(kdst, krBins, figsize=(14,10)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(kdst.full.Z, kdst.full.E, (krBins.Z, krBins.E))
    plt.colorbar().set_label("Number of events")
    labels("Z (mm)", "E (pes)", f" full ")

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(kdst.fid.Z, kdst.fid.E, (krBins.Z, krBins.E))
    plt.colorbar().set_label("Number of events")
    labels("Z (mm)", "E (pes)", f" fid ")

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(kdst.core.Z, kdst.core.E, (krBins.Z, krBins.E))
    plt.colorbar().set_label("Number of events")
    labels("Z (mm)", "E (pes)", f" core ")

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(kdst.hcore.Z, kdst.hcore.E, (krBins.Z, krBins.E))
    plt.colorbar().set_label("Number of events")
    labels("Z (mm)", "E (pes)", f" hard core Z")
    plt.tight_layout()

def plot_lifetime_T(kfs, timeStamps):
    ez0s = [kf.par[0] for kf in kfs]
    lts = [np.abs(kf.par[1]) for kf in kfs]
    u_ez0s = [kf.err[0] for kf in kfs]
    u_lts = [kf.err[1] for kf in kfs]
    plt.figure()
    ax=plt.gca()
    xfmt = md.DateFormatter('%d-%m %H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    plt.errorbar(timeStamps, lts, u_lts, fmt="kp", ms=7, lw=3)
    plt.xlabel('date')
    plt.ylabel('Lifetime (mus)')
    plt.xticks( rotation=25 )

def display_lifetime_maps(Escale : Measurement,
                          ELT: Measurement,
                          kltl : KrLTLimits,
                          XYcenters : np.array,
                          cmap = "jet",
                          mask = None):

    """Display lifetime maps: the mask allow to specify channels
    to be masked out (usually bad channels)
    """

    #fig = plt.figure(figsize=figsize)
    #fig.add_subplot(2, 2, 1)
    plt.subplot(2, 2, 1)
    *_, cb = display_matrix(XYcenters, XYcenters, Escale.value, mask,
                            vmin = kltl.Es.min,
                            vmax = kltl.Es.max,
                            cmap = cmap,
                            new_figure = False)
    cb.set_label("Energy scale at z=0 (pes)")
    labels("X (mm)", "Y (mm)", "Energy scale")

    #fig.add_subplot(2, 2, 2)
    plt.subplot(2, 2, 2)
    *_, cb = display_matrix(XYcenters, XYcenters, Escale.uncertainty, mask,
                        vmin = kltl.Eu.min,
                        vmax = kltl.Eu.max,
                        cmap = cmap,
                        new_figure = False)
    cb.set_label("Relative energy scale uncertainty (%)")
    labels("X (mm)", "Y (mm)", "Relative energy scale uncertainty")

    #fig.add_subplot(2, 2, 3)
    plt.subplot(2, 2, 3)
    *_, cb = display_matrix(XYcenters, XYcenters, ELT.value, mask,
                        vmin = kltl.LT.min,
                        vmax = kltl.LT.max,
                        cmap = cmap,
                        new_figure = False)
    cb.set_label("Lifetime (µs)")
    labels("X (mm)", "Y (mm)", "Lifetime")

    #fig.add_subplot(2, 2, 4)
    plt.subplot(2, 2, 4)
    *_, cb = display_matrix(XYcenters, XYcenters, ELT.uncertainty, mask,
                        vmin = kltl.LTu.min,
                        vmax = kltl.LTu.max,
                        cmap = cmap,
                        new_figure = False)
    cb.set_label("Relative lifetime uncertainty (%)")
    labels("X (mm)", "Y (mm)", "Relative lifetime uncertainty")

    plt.tight_layout()
