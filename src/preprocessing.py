from abc import ABC, abstractmethod

import numpy as np
from dataclasses import dataclass, field
from genericpath import isfile
from typing import Optional

import mne
import os
import logging

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
        return self.sp_read_ica_eeglab(ica_fname)

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

    def sp_read_ica_eeglab(self, fname, *, verbose=None):
        """Load ICA information saved in an EEGLAB .set file.
            Parameters
            ----------
            fname : str
                Complete path to a .set EEGLAB file that contains an ICA object.
            %(verbose)s

            Returns
            -------
            ica : instance of ICA
                An ICA object based on the information contained in the input file.
            """
        from scipy import linalg
        eeg = mne.preprocessing.ica._check_load_mat(fname, None)
        info, eeg_montage, _ = mne.preprocessing.ica._get_info(eeg)
        mne.pick_info(info, np.round(eeg['icachansind']).astype(int) - 1, copy=False)
        info.set_montage(eeg_montage)
        

        rank = eeg.icasphere.shape[0]
        n_components = eeg.icaweights.shape[0]

        ica = mne.preprocessing.ica.ICA(method='imported_eeglab', n_components=n_components)

        ica.current_fit = "eeglab"
        ica.ch_names = info["ch_names"]
        ica.n_pca_components = None
        ica.n_components_ = n_components

        n_ch = len(ica.ch_names)
        assert len(eeg.icachansind) == n_ch

        ica.pre_whitener_ = np.ones((n_ch, 1))
        ica.pca_mean_ = np.zeros(n_ch)

        assert eeg.icasphere.shape[1] == n_ch
        assert eeg.icaweights.shape == (n_components, rank)

        # When PCA reduction is used in EEGLAB, runica returns
        # weights= weights*sphere*eigenvectors(:,1:ncomps)';
        # sphere = eye(urchans). When PCA reduction is not used, we have:
        #
        #     eeg.icawinv == pinv(eeg.icaweights @ eeg.icasphere)
        #
        # So in either case, we can use SVD to get our square whitened
        # weights matrix (u * s) and our PCA vectors (v) back:
        use = eeg.icaweights @ eeg.icasphere
        use_check = linalg.pinv(eeg.icawinv)
        if not np.allclose(use, use_check, rtol=1e-6):
            logging.warn('Mismatch between icawinv and icaweights @ icasphere from EEGLAB '
                'possibly due to ICA component removal, assuming icawinv is '
                'correct')
            use = use_check
        u, s, v = mne.preprocessing.ica._safe_svd(use, full_matrices=False)
        ica.unmixing_matrix_ = u * s
        ica.pca_components_ = v
        ica.pca_explained_variance_ = s * s
        ica.info = info
        ica._update_mixing_matrix()
        ica._update_ica_names()
        return ica
