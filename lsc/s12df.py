import sys
import argparse

from typing import NamedTuple

import tables as tb
import numpy  as np
import pandas as pd


import invisible_cities.reco    .peak_functions          as pkf
import invisible_cities.reco    .calib_sensors_functions as csf
import invisible_cities.sierpe  .blr                     as blr
import invisible_cities.database.load_db                 as load_db

from   invisible_cities.evm  .ic_containers     import S12Params
from   invisible_cities.evm  .pmaps             import S1
from   invisible_cities.evm  .pmaps             import S2
from   invisible_cities.core .system_of_units_c import units
from   invisible_cities.core .exceptions        import SipmZeroCharge
from   invisible_cities.reco .xy_algorithms     import corona
from   invisible_cities.types.ic_types          import minmax
from   invisible_cities.types.ic_types          import NN


def create_empty_lists(number_of_lists = 2):
    return ([] for i in range(number_of_lists))


class Cluster(NamedTuple):
    Q     : float
    x     : float
    y     : float
    nsipm : int


class DummyS2(NamedTuple):
    total_energy       : float
    width              : float
    time_at_max_energy : float


def compute_xy_peak_position(sr, xs, ys):
        """
        Computes position using the integral of the charge
        in each SiPM. Config parameters set equal to the standard kDST values.
        """
        IDs    = sr.ids
        Qs     = sr.sum_over_times
        xs, ys = xs[IDs], ys[IDs]
        c      = corona(np.stack((xs, ys), axis=1), Qs,
                        Qthr           =  1,
                        Qlm            =  0,
                        lm_radius      = -1,
                        new_lm_radius  = -1,
                        msipm          =  1 )[0]

        return Cluster(Q=c.Q, x=c._xy.x, y=c._xy.y, nsipm=c.nsipm)


def s12df(run_number, filenames, mode='S1', event_range=NN, print_every=10):
    s12 = False   #S1 by default

    # Data base
    DataPMT         = load_db.DataPMT(run_number)
    DataSiPM        = load_db.DataSiPM(run_number)
    xs              = DataSiPM.X.values
    ys              = DataSiPM.Y.values
    pmt_active      = np.nonzero(DataPMT.Active.values)[0].tolist()
    channel_id      = DataPMT.ChannelID.values
    coeff_blr       = abs(DataPMT.coeff_blr.values)
    coeff_c         = abs(DataPMT.coeff_c .values)
    adc_to_pes      = abs(DataPMT.adc_to_pes.values)
    sipm_adc_to_pes = DataSiPM.adc_to_pes.values

    sipm_adc_to_pes[181] = 0
    adc_to_pes[9]        = 25

    #  S1 & S2 params
    s1params = S12Params(time          = minmax(min =   0,
                                                max = 620) * units.mus,
                         stride        = 4,
                         length        = minmax(min =  4,
                                                max = 25),  #250 ns -- 50 ns
                         rebin_stride  = 1)

    s2params = S12Params(time          = minmax(min = 640,
                                                max = 700) * units.mus,
                         stride        = 40,
                         length        = minmax(min = 80, max = 1e5),  # 4 mus --
                         rebin_stride  = 40)
    #  Tresholds
    s1th=   0.5 * units.pes
    s2th=   1.0 * units.pes
    sith = 10   * units.pes

    # DF init
    (evt,
     ns1, cs1, es1, hs1, ws1, ts1,
     ns2, es2, ws2, ts2,
     qs2, xs2, ys2, nsi          ) = create_empty_lists(number_of_lists = 15)

    # Loop
    event_number     = 0
    with_event_range = isinstance(event_range, minmax)

    for filename in filenames:
        with tb.open_file(filename, 'r') as file:
            pmtrwf  = file.root.RD.pmtrwf
            sipmrwf = file.root.RD.sipmrwf
            NEVT    = pmtrwf.shape[0]

            for event in range(NEVT):
                if with_event_range:
                    if event_number < event_range.min:
                        event_number += 1
                        continue
                    elif event_number > event_range.max:
                        break

                CWF            = blr.deconv_pmt(pmtrwf[event], coeff_c, coeff_blr,
                                                pmt_active, n_baseline  = 48000)
                _, _, cfsum, _ = csf.calibrate_pmts(CWF, adc_to_pes)

                sipm_cal       = csf.calibrate_sipms(sipmrwf[event], sipm_adc_to_pes, thr=sith,
                                                     bls_mode=csf.BlsMode.mode)

                if event_number % print_every == 0:
                    print(f"processing event = {event_number}")

                s1_indx, s1_ene = pkf.indices_and_wf_above_threshold(cfsum, s1th)

                if s1_indx.shape[0] == 0: continue

                s1s = pkf.find_peaks(cfsum,
                                     s1_indx,
                                     s1params.time,
                                     s1params.length,
                                     s1params.stride,
                                     s1params.rebin_stride,
                                     Pk=S1,
                                     pmt_ids=[-1],
                                     sipm_wfs=None,
                                     thr_sipm_s2=0)

                s2s = []
                s2  = DummyS2(total_energy = NN,
                              width        = NN,
                              time_at_max_energy = NN)

                c = Cluster(Q     = NN,
                            x     = NN,
                            y     = NN,
                            nsipm = NN)

                if mode == 'S12':
                    s2_indx, s2_ene = pkf.indices_and_wf_above_threshold(cfsum, s2th)
                    if s2_indx.shape[0] > 0:
                        s12 = True
                        s2s = pkf.find_peaks(cfsum,
                                             s2_indx,
                                             s2params.time,
                                             s2params.length,
                                             s2params.stride,
                                             s2params.rebin_stride,
                                             Pk=S2, pmt_ids=[-1],
                                             sipm_wfs=sipm_cal, thr_sipm_s2=0)
                        if len(s2s) > 0 :
                            s2 = s2s[0]
                            try:
                                c = compute_xy_peak_position(s2.sipms, xs, ys)
                            except SipmZeroCharge:
                                c = Cluster(Q     = NN,
                                            x     = NN,
                                            y     = NN,
                                            nsipm = NN)

                ns1.append(len(s1s))
                ns2.append(len(s2s))

                if s12:
                    if len(s1s) == 0 or len(s2s) == 0 : continue


                for s1 in s1s:
                    evt.append(event_number)
                    cs1.append(len(s1s))
                    es1.append(s1.total_energy)
                    hs1.append(s1.height)
                    ws1.append(s1.width)
                    ts1.append(s1.time_at_max_energy/units.mus)
                    es2.append(s2.total_energy)
                    ws2.append(s2.width)
                    ts2.append(s2.time_at_max_energy/units.mus)
                    qs2.append(c.Q)
                    xs2.append(c.x)
                    ys2.append(c.y)
                    nsi.append(c.nsipm)

                event_number +=1


    pdf = pd.DataFrame(np.array([evt, cs1, es1, hs1, ws1, ts1,
                                           es2,      ws2, ts2,
                                           qs2, xs2, ys2, nsi]).T,
                       columns= ['evt', 'ns1', 'es1', 'hs1', 'ws1', 'ts1',
                                               'es2',        'ws2', 'ts2',
                                               'qs2', 'xs2', 'ys2', 'nsi'])

    return np.array(ns1), np.array(ns2), pdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", type=str, nargs="*"              )
    parser.add_argument("-o"       , type=str                         )
    parser.add_argument("-r"       , type=int                         )
    parser.add_argument("--mode"   , type=str,            default="S1")
    parser.add_argument("-p"       , type=int, nargs="*", default= 100)

    args = parser.parse_args(sys.argv[1:])
    *_, df   = s12df(args.r, args.filenames, args.mode, NN, args.p)
    df.to_hdf(args.o,
              key     = "DST"  , mode         = "w",
              format  = "table", data_columns = True,
              complib = "zlib" , complevel    = 4)

    with tb.open_file(args.o, "r+") as file:
        file.rename_node(file.root.DST.table, "Events")
        file.root.DST.Events.title          = "Events"
