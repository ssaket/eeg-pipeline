import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import ccs_eeg_utils
from mne_bids import (write_raw_bids, BIDSPath, read_raw_bids, print_dir_tree)

bids_root = os.path.join('data', 'P3')
bids_path = BIDSPath(subject='001', session='P3', task='P3',
                     datatype='eeg', suffix='eeg', root=bids_root)

raw = read_raw_bids(bids_path=bids_path)
events, event_id = mne.events_from_annotations(raw)
# ccs_eeg_utils.read_annotations_core(bids_path,raw)

raw.load_data()
# raw.pick_channels(["Pz"])  # ["Pz","Fz","Cz"])
# plt.plot(raw[:, :][0].T)
# raw.plot_psd(area_mode='range', tmax=10.0, average=False, xscale="linear",)

raw.plot_psd(tmax=np.inf, fmax=250)
raw_f = raw.copy().filter(0.1,50, fir_design='firwin')
raw_f.plot_psd(tmax=np.inf, fmax=250)

# raw.resample(100, npad="auto")  # set sampling frequency to 100Hz
# raw.plot_psd(area_mode='range', tmax=10.0, picks=picks)

raw.plot(n_channels=len(raw.ch_names))