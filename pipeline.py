from warnings import simplefilter
import pandas as pd
from dataclasses import dataclass
from genericpath import isfile
from mne_bids import (BIDSPath, read_raw_bids)
from typing import Optional, Union
from preprocessing import *

import os
import mne
import logging

logging.basicConfig(level=logging.INFO)
mne.set_log_level(verbose='ERROR')
simplefilter(action='ignore', category=FutureWarning)


@dataclass
class Pipeline:
    """ Pipeline for processing Encoding and Decoding Analysis on EEG data"""
    bids_path: Union[str, list[str]]
    subject: Optional[int] = None
    raw: mne.io.Raw = field(init=False, repr=False)
    events: np.ndarray = field(init=False, repr=False)
    event_ids: dict = field(init=False, repr=False)

    def set_montage(self) -> None:
        montage_dir = os.path.join(os.path.dirname(mne.__file__),
                                   'channels', 'data', 'montages')
        logging.debug(sorted(os.listdir(montage_dir)))
        ten_twenty_montage = mne.channels.make_standard_montage(
            'standard_1020')
        logging.info("Setting montage")
        self.raw.set_montage(ten_twenty_montage, match_case=False)

    def load_data(self) -> None:
        logging.info("Loading Data")
        raw = read_raw_bids(bids_path=self.bids_path)
        self.events, self.event_ids = mne.events_from_annotations(raw)
        self.raw = raw.load_data()

    def apply_resampling(self, sampling_freq: int, padding: str = 'auto') -> None:
        logging.info("Applying resampling")
        self.raw.resample(sampling_freq, npad=padding)

    def apply_rereferencing(self, reference_channels: List[str]) -> None:
        logging.info("Applying re-referencing")
        self.raw.set_eeg_reference()

    def apply_cleaning(self, cleaner: CleaningData):
        logging.info("Applying cleaning")
        cleaner.apply_cleaning(self.raw)

    def apply_filter(self, filter: BaseFilter) -> None:
        logging.info("Applying filtering")
        filter.apply_filter(self.raw)

    def apply_ica(self, ica: BaseICA) -> None:
        logging.info("Applying ICA")
        ica.compute_ica(self.raw)
        ica.apply_ica(self.raw)

    def get_events_df(self, events_ext: str = 'events.tsv') -> pd.DataFrame:
        bids_path = self.bids_path
        fname = os.path.join(bids_path.directory, bids_path.basename.removesuffix(
            bids_path.suffix) + events_ext)
        assert isfile(fname), "Events file not found!"
        return pd.read_csv(fname, delimiter='\t')

    def compute_erp_peak(self, erp: ERPPeak, condition: str) -> pd.DataFrame:
        erp.compute_epochs(self.raw, self.events, self.event_ids)
        return erp.compute_peak(condition)

    def start_preprocessing(self):
        logging.info(
            "*"*5 + "Proceesing for subject: {}". format(self.bids_path.subject) + "*"*5)
        self.load_data()
        self.set_montage()
        self.apply_cleaning(CleaningData(self.bids_path))
        self.apply_filter(SimpleMNEFilter(0.1, 50, 'firwin'))
        self.apply_ica(PrecomputedICA(self.bids_path))
        self.compute_erp_peak(ERPPeak(-0.2, 0.8, (-0.2, 0)))
        logging.info("Processed subject {}\n".format(self.bids_path.subject))


def load_all_subjects(bids_path: BIDSPath):
    # pre_ica = PrecomputedICA(bids_path)
    # pre_ica.compute_ica()
    pip = Pipeline(bids_path)
    try:
        pip.start_preprocessing()
        return "done"
    except Exception as err:
        logging.error("failed for subject {}".format(bids_path.subject))
        logging.error(err)
        return bids_path.subject

    # clean = CleaningData(bids_path)
    # clean.load_bad_data()

    # return clean.bad_annotations


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
    bids_paths = [bids_path.copy().update(subject=str(x).zfill(3))
                  for x in range(1, 41)]
    load_all_subjects(bids_paths[1])
    # a = map(load_all_subjects, bids_paths)
    # print(list(a))
