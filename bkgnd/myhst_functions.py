#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:21:59 2017
@author: hernando
"""

import numpy             as np
import scipy.stats       as stats
import matplotlib.pyplot as plt
import invisible_cities.icaro.hst_functions as hst
from invisible_cities.icaro.hst_functions import shift_to_bin_centers
from invisible_cities.icaro.hst_functions import resolution, gausstext
from invisible_cities.icaro.hst_functions import labels

class Canvas:
    
    def __init__(self, nx, ny, nxsize=5.4, nysize=6.2):
        self.nx = nx
        self.ny = ny
        self.n  = 1
        plt.figure(figsize=(nysize*ny, nxsize*nx))
        plt.subplot(nx, ny, self.n)
        
    def __call__(self, i):
        plt.subplot(self.nx, self.ny, i)
        return True
        
    def __add__(self, i):
        self.n += i
        plt.subplot(self.nx, self.ny, n)
        return True


def _xypos(xs, ys, xf=0.1, yf=0.7):
    x0, dx = min(xs), max(xs)-min(xs)
    y0, dy = min(ys), max(ys)-min(ys)
    xp = x0 + xf*dx
    yp = y0 + yf*dy
    return (xp, yp)

def _hist(xs, bins=100, range=None, 
         stats=True, xylabels = (), stats_xypos=(0.1, 0.7),
         *args, **kargs):
    """
    hist function extended to plot stattistics 
    inputs:
        stats: bool (default=True)
            write stats in the plot
        stats_xypos: tuple (x, y) (default (0.1, 0.7))
            (x, y) relative positions in the plot, 
            x, y values in rage (0., 1.)
    outputs:
        none
    """        
    if (range==None):
        range = (np.min(xs), np.max(xs))
    cc = hst.hist(xs, bins=bins, range=range, *args, **kargs);
    if (not stats):
        return cc
    ys, xedges = np.histogram(xs, bins, range=range)
    ns = len(xs)
    sel = np.logical_and(xs >= range[0], xs <= range[1])
    nos, mean, rms = len(xs[sel]), np.mean(xs[sel]), np.std(xs[sel])
    epsilon = (1.*nos)/(1.*ns)
    ss  = ' $\epsilon$ = {0:.3f} \n $n$ = {1:}'.format(epsilon, nos)
    ss += ' \n $\mu$ = {0:.3f} \n $\sigma$ = {1:.3f}'.format(mean, rms)
    xp, yp = _xypos(xedges, ys, xf=stats_xypos[0], yf=stats_xypos[1])
    plt.text(xp, yp, ss)
    return cc


def _decorate(fun):
    def _decorator(*args, **kargs):
        if ('canvas' in kargs):
            kargs['new_figure'] = False
            del kargs['canvas']
        has_xylabels, xylabels = 'xylabels' in kargs, ()
        if (has_xylabels):
            xylabels = kargs['xylabels']
            del kargs['xylabels']
        response = fun(*args, **kargs)
        if (len(xylabels)>0):
            hst.labels(*xylabels)
        return response 
    return _decorator

hist                = _decorate(_hist)
plot                = _decorate(hst.plot)
hist2d              = _decorate(hst.hist2d)
errorbar            = _decorate(hst.errorbar)
pdf                 = _decorate(hst.pdf)
scatter             = _decorate(hst.scatter)
profile_and_scatter = _decorate(hst.profile_and_scatter)
hist2d_profile      = _decorate(hst.hist2d_profile)
display_matrix      = _decorate(hst.display_matrix)
