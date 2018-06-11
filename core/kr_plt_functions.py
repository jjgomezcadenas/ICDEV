import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from invisible_cities.icaro.mpl_functions import set_plot_labels
from   invisible_cities.core.system_of_units_c import units

from . kr_types import KrLTLimits
from invisible_cities.icaro. hst_functions import display_matrix
from   invisible_cities.evm  .ic_containers  import Measurement
from invisible_cities.icaro. hst_functions import display_matrix
from icaro.core.fit_functions import conditional_labels

labels = conditional_labels(True)


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
