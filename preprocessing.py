from abc import ABC, abstractmethod

import numpy as np
from numpy.core.fromnumeric import argmax
from ccs_eeg_semesterproject import sp_read_ica_eeglab
from dataclasses import dataclass, field
from genericpath import isfile
from typing import Dict, List, Optional, Tuple, Union

import mne
import os
import logging
import pandas as pd

from mne_bids.path import BIDSPath


@dataclass
class BaseFilter(ABC):
    @abstractmethod
    def apply_filter():
        pass
    @staticmethod
    def step():
        return "filtering"


@dataclass
class SimpleMNEFilter(BaseFilter):
    """ Simple filter based on MNE. """
    l_freq: float
    h_freq: float
    name: str

    def apply_filter(self, raw: mne.io.Raw):
        return raw.filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design=self.name)


@dataclass
class BaseICA(ABC):
    @abstractmethod
    def compute_ica(self, raw: mne.io.Raw):
        pass
    @abstractmethod
    def apply_ica(self, raw: mne.io.Raw, make_copy: bool = False):
        pass
    @staticmethod
    def step():
        return "ica"


@dataclass
class SimpleMNEICA(BaseICA):
    """ Simple ICA based on MNE """
    method: str
    n_components: Optional[int] = None
    random_state: int = 23
    exclude: list[int] = field(default_factory=list)
    ica: mne.preprocessing.ica = field(init=False)

    def compute_ica(self, raw: mne.io.Raw):
        self.ica = mne.preprocessing.ICA(
            n_components=self.n_components, method=self.method, random_state=self.random_state)
        self.ica.fit(raw, verbose=True)

    def apply_ica(self, raw: mne.io.Raw, make_copy: bool = False):
        self.ica.exclude = self.exclude
        if make_copy:
            return self.ica.apply(raw.copy())
        else:
            return self.ica.apply(raw)


"""
Classes for loading preprocessed and precomputed EEG data provided by the CCS-Department.
"""


@dataclass
class CleaningData():
    """Load precomputed bad channels and segments provided by the CCS-Department"""
    bids_path: BIDSPath
    bad_channels: list[int] = field(default_factory=list)
    bad_annotations: list = field(default_factory=list, repr=False)
    channels_ext: str = 'badChannels.tsv'
    segments_ext: str = 'badSegments.csv'
    
    @staticmethod
    def step():
        return "cleaning"

    def _get_fpath(self, dtype: str):
        assert dtype in [
            'channels', 'segments'], "dataType can only have values from ['channels', 'segments']"
        bids_path = self.bids_path
        etx = self.channels_ext if dtype == 'channels' else self.segments_ext
        bad_fname = os.path.join(bids_path.directory, bids_path.basename.removesuffix(
            bids_path.suffix) + etx)
        assert isfile(bad_fname), "Bad {} file not found!".format(dtype)
        return bad_fname

    def _get_bad_channels(self) -> np.ndarray:
        ch_fname = self._get_fpath('channels')
        bad_channels = np.loadtxt(ch_fname, delimiter='\t', dtype='int')
        bad_channels -= 1  # handle 0 indexing
        return bad_channels.reshape(-1)

    def _get_bad_segments(self) -> mne.Annotations:
        import pandas as pd
        seg_fname = self._get_fpath('segments')
        df = pd.read_csv(seg_fname)
        return mne.Annotations(df.onset, df.duration, df.description)

    def load_bad_data(self):
        self.bad_annotations = self._get_bad_segments()
        self.bad_channels = self._get_bad_channels()

    def apply_cleaning(self, raw: mne.io.Raw, interpolate: bool = True):
        self.load_bad_data()
        raw.info['bads'] = [raw.ch_names[idx] for idx in self.bad_channels]
        if interpolate:
            raw.interpolate_bads()
        raw.set_annotations(raw.annotations + self.bad_annotations)


@dataclass
class PrecomputedICA(BaseICA):
    """Load precomputed ICA provided by the CCS-Department"""
    bids_path: BIDSPath
    ica_ext: str = 'ica.set'
    badComponent_ext: str = 'ica.tsv'
    ica: mne.preprocessing.ica = field(init=False)
    exclude: list[int] = field(default_factory=list)

    def _load_bad_components(self, badComponents_fname: str, delimiter: str = '\t') -> np.ndarray:
        assert isfile(
            badComponents_fname), "ICA Bad Components file not found!"
        bad_components = np.loadtxt(
            badComponents_fname, delimiter=delimiter, dtype='float')
        # for zero indexing
        bad_components -= 1
        # for cases when we have only one component in the file
        return bad_components.reshape(-1)

    def _load_ica(self, ica_fname: str) -> mne.preprocessing.ica:
        assert isfile(ica_fname), "ICA file not found!"
        return sp_read_ica_eeglab(ica_fname)

    def compute_ica(self, raw: mne.io.Raw = None):
        bids_path = self.bids_path
        ica_file = os.path.join(bids_path.directory, bids_path.basename.removesuffix(
            bids_path.suffix) + self.ica_ext)
        self.ica = self._load_ica(ica_file)
        bad_comp = os.path.join(bids_path.directory, bids_path.basename.removesuffix(
            bids_path.suffix) + self.badComponent_ext)
        self.exclude = self._load_bad_components(bad_comp)

    def apply_ica(self, raw: mne.io.Raw, make_copy: bool = False):
        self.ica.exclude = self.exclude
        if make_copy:
            self.ica.apply(raw.copy())
        else:
            self.ica.apply(raw)
