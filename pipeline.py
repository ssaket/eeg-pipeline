from erpanalysis import ERPAnalysis

from tqdm import tqdm
from multiprocessing import Pool
from dataclasses import dataclass
from genericpath import isfile
from mne_bids import (BIDSPath, read_raw_bids)
from typing import Optional, Union
from preprocessing import *

import os
import mne
import logging
import warnings
import pandas as pd


def P3_EVENTS_MAPINGS() -> Tuple[dict, list[int], list[int]]:
    blocks = np.array([list(range(10*x + 1, 10*x + 6)) for x in range(1, 6)])
    rare = np.array([x + i for i, x in enumerate(range(11, 56, 10))]).tolist()
    freq = np.setdiff1d(blocks.flatten(), rare).tolist()

    stimlus = ['A', 'B', 'C', 'D', 'E']

    evts_stim = ['stimulus/' + stimlus[i] + '/' +
                 str(alph) for i, x in enumerate(blocks) for alph in x]
    evts_id = dict((i+3, evts_stim[i]) for i in range(0, len(evts_stim)))
    evts_id[1] = 'response/201'
    evts_id[2] = 'response/202'
    return evts_id, rare, freq


@dataclass
class Pipeline:
    """ Pipeline for processing Encoding and Decoding Analysis on EEG data"""
    bids_path: Union[str, list[str]]
    subject: Optional[int] = None
    verbose: logging = logging.INFO
    raw: mne.io.Raw = field(init=False, repr=False)
    events: np.ndarray = field(init=False, repr=False)
    event_ids: dict = field(init=False, repr=False)

    def __post_init__(self):
        logging.basicConfig(level=self.verbose)
        mne.set_log_level(verbose='ERROR')
        warnings.filterwarnings("ignore")

    def set_montage(self) -> None:
        montage_dir = os.path.join(os.path.dirname(mne.__file__),
                                   'channels', 'data', 'montages')
        logging.debug(sorted(os.listdir(montage_dir)))
        ten_twenty_montage = mne.channels.make_standard_montage(
            'standard_1020')
        logging.info("Setting montage")
        self.raw.set_montage(ten_twenty_montage, match_case=False)

    def load_data(self, event_id: Union[Dict, str] = "auto") -> None:
        logging.info("Loading Data")
        raw = read_raw_bids(bids_path=self.bids_path)
        self.events, self.event_ids = mne.events_from_annotations(
            raw, event_id=event_id)
        self.raw = raw.load_data()

    def set_custom_events_mapping(self, mapping: Dict[int, str] = None, task: str = None) -> None:
        if task == 'P3':
            mapping, _, _ = P3_EVENTS_MAPINGS()
        assert mapping is not None, "Mapping is not defined! Please pass mapping as argument"

        annot_from_events = mne.annotations_from_events(
            events=self.events, event_desc=mapping, sfreq=self.raw.info['sfreq'])
        self.raw.set_annotations(annot_from_events)
        self.events, self.event_ids = mne.events_from_annotations(self.raw)

    def apply_resampling(self, sampling_freq: int, padding: str = 'auto') -> None:
        logging.info("Applying resampling")
        self.raw.resample(sampling_freq, npad=padding)

    def apply_rereferencing(self, reference_channels: Union[List[str], str]) -> None:
        logging.info("Applying re-referencing")
        mne.io.set_eeg_reference(self.raw, ref_channels=reference_channels)

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

    def compute_epochs(self, erp: ERPAnalysis) -> mne.Epochs:
        return erp.compute_epochs(self.raw, self.events, self.event_ids)

    def compute_erp_peak(self, erp: ERPAnalysis, condition: str, thypo: float, offset: float = 0.05, channels: list[str] = []) -> pd.DataFrame:
        self.compute_epochs(erp)
        return erp.compute_peak(condition, thypo, offset, channels)
    
    def _parallel_process(self, pipeline):
        pipeline.load_data()
        pipeline.set_montage()
        pipeline.make_pipeline([CleaningData(pipeline.bids_path), SimpleMNEFilter(0.1, 50, 'firwin'), PrecomputedICA(pipeline.bids_path)])
        pipeline.set_custom_events_mapping(task='P3')
        return pipeline.raw

    def load_multiple_subjects(self, n_subjects=40, preload: bool = False) -> None:
        
        curr_sub = [int(self.bids_path.subject)]
        subjects = set(range(1, n_subjects + 1)) - set(curr_sub)
        bids_paths = [self.bids_path.copy().update(subject=str(x).zfill(3))
                  for x in subjects]
        pipelines = [ Pipeline(bids_path=path, verbose=logging.ERROR) for path in bids_paths ]
        with Pool(6) as p:
            raws = list(tqdm(p.imap(self._parallel_process, pipelines), total=n_subjects-1))
            
        
        raws.append(self.raw)
        self.raw = mne.concatenate_raws(raws)
        self.events, self.event_ids = mne.events_from_annotations(self.raw)
        if preload: self.raw.load_data()
        self.set_montage()

    def apply(self, step):
        if step.step() == 'cleaning':
            self.apply_cleaning(step)
        elif step.step() == 'filtering':
            self.apply_filter(step)
        elif step.step() == 'ica':
            self.apply_ica(step)
        elif step.step() == 'erp':
            self.compute_epochs(step)
        else:
            logging.error("Invalid pipeline operation!")

    def make_pipeline(self, steps: list):
        logging.info(
            "*"*5 + "Proceesing for subject: {}". format(self.bids_path.subject) + "*"*5)
        for step in steps:
            self.apply(step)
        logging.info("Processed subject {}\n".format(self.bids_path.subject))


def load_all_subjects(bids_path: BIDSPath):
    # pre_ica = PrecomputedICA(bids_path)
    # pre_ica.compute_ica()
    pip= Pipeline(bids_path)
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

    bids_root= os.path.join('data', 'P3')
    bids_path= BIDSPath(subject='030', session='P3', task='P3',
                         datatype='eeg', suffix='eeg', root=bids_root)

    # pip = Pipeline(bids_path)
    # pip.load_data()

    # sim_filter = SimpleMNEFilter(0.1, 50, 'firwin')
    # pip.apply_filter(sim_filter)

    # pre_ica = PrecomputedICA(bids_path)
    # pre_ica.compute_ica()
    bids_paths= [bids_path.copy().update(subject=str(x).zfill(3))
                  for x in range(1, 41)]
    load_all_subjects(bids_paths[1])
    # a = map(load_all_subjects, bids_paths)
    # print(list(a))
