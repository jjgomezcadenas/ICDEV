import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from invisible_cities.icaro.mpl_functions import set_plot_labels
from   invisible_cities.core.system_of_units_c import units

def plot_S12(s12s):
    plt.grid(True)
    plt.xlabel(r't (ns)')
    plt.ylabel(r'q (pes)')
    for s12 in s12s:
        plt.plot(s12.times, s12.pmts.waveform(-1))

def plot_pmt_signals_vs_time_mus(pmt_signals,
                                 pmt_active,
                                 t_min      =    0,
                                 t_max      = 1200,
                                 signal_min =    0,
                                 signal_max =  200,
                                 figsize=(12,12)):
    """Plot PMT signals versus time in mus  and return figure."""

    tstep = 25
    PMTWL = pmt_signals[0].shape[0]
    signal_t = np.arange(0., PMTWL * tstep, tstep)/units.mus
    plt.figure(figsize=figsize)


    plt.ylabel(r'q (pes/adc)')
    for j, i in enumerate(pmt_active):
        plt.grid(True)
        ax1 = plt.subplot(3, 4, j+1)
        ax1.set_xlim([t_min, t_max])
        ax1.set_ylim([signal_min, signal_max])
        plt.plot(signal_t, pmt_signals[i])
        plt.xlabel(r't (mus)')



def plot_cwf_vs_time_mus(signal,
                         t_min      =    0,
                         t_max      = 1300,
                         t_trg      = 650,
                         t_trgw     = 50,
                         s2_min =    0,
                         s2_max =  200,
                         s1_min =    0,
                         s1_max =  10,
                         figsize=(10,10)):

    plt.figure(figsize=figsize)
    plt.grid(True)

    tstep = 25 # in ns
    PMTWL = signal.shape[0]
    signal_t = np.arange(0., PMTWL * tstep, tstep)/units.mus

    ax1 = plt.subplot(2, 2, 1)
    ax1.set_xlim([t_min, t_max])
    ax1.set_ylim([s2_min, s2_max])
    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "signal (pes/adc)")
    plt.plot(signal_t, signal)

    ax1 = plt.subplot(2, 2, 2)
    ax1.set_xlim([t_trg - t_trgw, t_trg + t_trgw])
    ax1.set_ylim([s2_min, s2_max])
    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "signal (pes/adc)")
    plt.plot(signal_t, signal)

    ax1 = plt.subplot(2, 2, 3)
    ax1.set_xlim([t_min,  t_trg - t_trgw])
    ax1.set_ylim([s1_min, s1_max])
    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "signal (pes/adc)")
    plt.plot(signal_t, signal)

    ax1 = plt.subplot(2, 2, 4)
    ax1.set_xlim([t_trg + t_trgw,  t_max])
    ax1.set_ylim([s1_min, s1_max])
    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "signal (pes/adc)")
    plt.plot(signal_t, signal)


def plot_pmt_waveforms(pmtwfdf, first=0, last=50000, figsize=(10,10)):
    """plot PMT wf and return figure"""
    plt.figure(figsize=figsize)
    for i in range(len(pmtwfdf)):
        ax = plt.subplot(3, 4, i+1)
        plt.xlabel(r't (mus)')
        plt.plot(pmtwfdf[i][first:last])


def plot_sipm_map(sipm_cal, xs, ys, t_min=0, t_max=1300, zoom=False):

    sipm_sums_w = np.sum(sipm_cal[:, t_min:t_max], axis=1)

    plt.figure()
    sipms_ = sipm_sums_w > 0
    x = xs[sipms_]
    y = ys[sipms_]
    q = sipm_sums_w[sipms_]
    print(np.argwhere(sipms_))
    plt.scatter(x, y, s=10, c=q, cmap="jet")

    if not zoom:
        plt.xlim(-200, 200)
        plt.ylim(-200, 200)
    plt.colorbar()
