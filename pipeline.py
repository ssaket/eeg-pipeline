from dataclasses import dataclass
from mne_bids import (BIDSPath, read_raw_bids)
from typing import List, Optional
from preprocessing import BaseFilter, CleaningData, PrecomputedICA, SimpleMNEFilter

import os
import mne


@dataclass
class Pipeline:
    """ Pipeline for processing Encoding and Decoding Analysis on EEG data"""
    bids_path: List[str]
    subject: Optional[int] = None

    def set_montage(self) -> None:
        montage_dir = os.path.join(os.path.dirname(mne.__file__),
                                   'channels', 'data', 'montages')
        print('\nBUILT-IN MONTAGE FILES')
        print('======================')
        print(sorted(os.listdir(montage_dir)))
        ten_twenty_montage = mne.channels.make_standard_montage(
            'standard_1020')
        self.raw.set_montage(ten_twenty_montage, match_case=False)

    def load_data(self) -> None:
        raw = read_raw_bids(bids_path=self.bids_path)
        events, event_id = mne.events_from_annotations(raw)
        self.raw = raw.load_data()
        self.events = events
        self.event_id = event_id

    def apply_resampling(self, sampling_freq: int, padding: str = 'auto') -> None:
        self.raw.resample(sampling_freq, npad=padding)

    def apply_rereferencing(self, reference_channels: List[str]) -> None:
        self.raw.set_eeg_reference()

    def apply_filter(self, filter: BaseFilter) -> None:
        filter.apply_filter(self.raw)

    def compute_ica(slef) -> None:
        pass

    def compute_erp_peak(self) -> None:
        pass
    
def load_all_subjects(bids_path: BIDSPath):
    # pre_ica = PrecomputedICA(bids_path)
    # pre_ica.compute_ica()
    
    clean = CleaningData(bids_path)
    clean.load_bad_data()
    
    return clean.bad_annotations
    


if __name__ == '__main__':

    bids_root = os.path.join('data', 'P3')
    bids_path = BIDSPath(subject='001', session='P3', task='P3',
                         datatype='eeg', suffix='eeg', root=bids_root)

    # pip = Pipeline(bids_path)
    # pip.load_data()

    # sim_filter = SimpleMNEFilter(0.1, 50, 'firwin')
    # pip.apply_filter(sim_filter)
    
    # pre_ica = PrecomputedICA(bids_path)
    # pre_ica.compute_ica()
    bids_paths = [ bids_path.copy().update(subject=str(x).zfill(3)) for x in range(1, 41)]
    # load_all_subjects(bids_paths[0])
    a = map(load_all_subjects, bids_paths)
    print(list(a))
    
