import numpy as np

import matplotlib.pyplot as plt
from   invisible_cities.core.core_functions import weighted_mean_and_var
from   invisible_cities.core.core_functions import loc_elem_1d



def ns1(nsdf):
    hns1, bins  = histo_ns12(nsdf.ns1)
    print(' 0S1/tot  = {} 1S1/tot = {} 2S1/tot = {}'.format(bin_ratio(hns1, bins, 0),
                                                            bin_ratio(hns1, bins, 1),
                                                            bin_ratio(hns1, bins, 2)))



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
    return hns1, bins


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
