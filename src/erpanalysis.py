from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import mne
import logging
from scipy.stats.mstats import winsorize


@dataclass
class ERPAnalysis():
    """ERP Peak Analysis"""
    tmin: float
    tmax: float
    baseline: Optional[Tuple[Optional[int], int]] = None
    reject_by_annotation: bool = False
    all_subjects: bool = False
    epochs: Union[mne.Epochs, list[mne.Epochs]] = field(init=False,
                                                        repr=False,
                                                        default_factory=list)
    reject: Dict = field(init=False, default_factory=dict)

    @staticmethod
    def step():
        return "erp"

    def compute_epochs(self,
                       raw: mne.io.Raw,
                       events: np.ndarray,
                       events_id: Dict,
                       set_default: bool = True,
                       **kwargs) -> Union[None, mne.Epochs]:
        """Compute and Return the epochs """
        baseline = kwargs.pop('baseline', self.baseline)
        reject_by_annotation = kwargs.pop('reject_by_annotation',
                                          self.reject_by_annotation)
        reject = kwargs.pop('reject', self.reject)
        epochs = mne.Epochs(raw,
                            events=events,
                            event_id=events_id,
                            tmin=self.tmin,
                            tmax=self.tmax,
                            baseline=baseline,
                            reject_by_annotation=reject_by_annotation,
                            reject=reject,
                            preload=True)

        if self.all_subjects:
            self.epochs.append(epochs)
            return
        if set_default:
            self.epochs = epochs

        return epochs

    def get_peak_channel(self, trial: mne.Evoked, channel: str, tmin: float,
                         tmax: float,
                         mode: str) -> Tuple[str, float, float, float]:
        """
        Return the peak amplitude and mean amplitude channel voltage
        
        The mean is calculated by winsorize window removing 30% of the larger and smaller values in the data to minimize the influence of outliers.
        """

        trial = trial.pick(channel)
        data = trial.data

        tmin_idx, tmax_idx = trial.time_as_index([tmin, tmax],
                                                 use_rounding=True)
        data = data[:, tmin_idx:tmax_idx]

        if mode == 'abs':
            data = np.absolute(data)[0]

        return (channel[np.argmax(data) // data.shape[1]],
                trial.times[tmin_idx + (np.argmax(data) % data.shape[1])],
                np.max(data), np.mean(winsorize(data, limits=[0.3, 0.3])))

    def compute_peak(self,
                     stim: str,
                     tmin: float,
                     tmax: float,
                     channels: list[str],
                     mode: str = 'pos') -> pd.DataFrame:
        """Computes and returns the ERP peak values as Pandas Dataframes"""

        assert type(self.epochs) != np.ndarray, "Run compute_epochs first!"
        if isinstance(self.epochs, list):
            erp_dfs = []
            for _i, epoch in enumerate(self.epochs):
                erp_dfs.append(
                    self._get_erp_df(epoch, stim, tmin, tmax, channels, mode))
            return erp_dfs
        else:
            return self._get_erp_df(self.epochs, stim, tmin, tmax, channels,
                                    mode)

    def _get_erp_df(self, epochs: mne.Epochs, stim: str, tmin: float,
                    tmax: float, channels: list[str],
                    mode: str) -> pd.DataFrame:
        """ Helper function to calculate peak values based on given time 't' padded by the offset values """
        epochs.load_data()
        peak_values = {
            'channel': [],
            'peak_amp': [],
            'mean_amp': [],
            'latency': [],
            'trial': [],
            'stimulus': [],
            'condition': []
        }
        _epochs: mne.Epochs = epochs[stim] if stim else epochs

        for ix, trial in enumerate(_epochs.iter_evoked()):

            _channel, _latency, _peak = trial.get_peak(tmin=tmin,
                                                       tmax=tmax,
                                                       ch_type='eeg',
                                                       return_amplitude=True,
                                                       mode=mode)

            # We are using because MNE does not support mean amplitude using time window
            # and neither channel selection
            channel, latency, peak, mamp = self.get_peak_channel(trial,
                                                                 channels,
                                                                 tmin,
                                                                 tmax,
                                                                 mode=mode)
            # to make sure that our logic is correct
            if _channel == channel:
                # allow difference upto 2 µV and latency upto 5 milliseconds
                assert round(abs(peak - _peak) *
                             1e6) < 2, "Incorrect peak calculation logic!"
                assert round(abs(latency - _latency) *
                             1e3) < 5, "Incorrect latency calculation logic!"

            latency = int(round(latency * 1e3))  # convert to milliseconds
            peak = int(round(peak * 1e6))  # convert to µV
            mamp = int(round(mamp * 1e6))  # convert to µV
            peak_values['channel'].append(channel)
            peak_values['peak_amp'].append(peak)
            peak_values['mean_amp'].append(mamp)
            peak_values['latency'].append(latency)
            peak_values['trial'].append(ix)
            if len(stim.split('/')) > 1:
                peak_values['stimulus'].append(stim.split('/')[0])
                peak_values['condition'].append(stim.split('/')[1])
            else:
                peak_values['stimulus'].append('stimulus')
                peak_values['condition'].append(stim)

            logging.debug(
                'Trial {}: peak of {} µV and mean of {} µV at {} ms in channel {}'
                .format(ix, peak, mamp, latency, channel))
        df = pd.DataFrame.from_dict(peak_values)
        del peak_values
        return df
