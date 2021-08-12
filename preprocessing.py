from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from __future__ import annotations

import mne


class BaseFilter(ABC):
    @abstractmethod
    def apply_filter():
        pass


@dataclass
class SimpleMNEFilter(BaseFilter):
    """ Simple filter based on MNE. """
    h_freq: int
    l_freq: int
    name: str

    def apply_filter(self, raw: mne.io.Raw):
        raw.filter(self.l_freq, self.h_freq, fir_design=self.name)


class BaseICA(ABC):
    @abstractmethod
    def compute_ica():
        pass

    @abstractmethod
    def apply_ica():
        pass


@dataclass
class SimpleMNEICA(BaseICA):
    """ Simple ICA based on MNE"""
    method: str
    exclude: List[int] = []
    n_components: Optional[int] = None
    random_state: Optional[int] = 23

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
