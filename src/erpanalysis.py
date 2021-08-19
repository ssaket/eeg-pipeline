from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import mne
import logging


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

    @staticmethod
    def step():
        return "erp"

    def compute_epochs(self,
                       raw: mne.io.Raw,
                       events: np.ndarray,
                       events_id: Dict,
                       set_default: bool = True,
                       **kwargs) -> Union[None, mne.Epochs]:
        """There is a bug in MNE, even if we pass the default baseline (None, 0), we get slightly different results."""
        baseline = kwargs.pop('baseline', self.baseline)
        reject_by_annotation = kwargs.pop('reject_by_annotation',
                                          self.reject_by_annotation)
        epochs = mne.Epochs(raw,
                            events=events,
                            event_id=events_id,
                            tmin=self.tmin,
                            tmax=self.tmax,
                            baseline=baseline,
                            reject_by_annotation=reject_by_annotation,
                            **kwargs)

        if self.all_subjects:
            self.epochs.append(epochs)
            return
        if set_default:
            self.epochs = epochs

        return epochs

    def get_peak_channel(
            self,
            trial: mne.Evoked,
            channel: str,
            tmin: float,
            tmax: float,
            average: bool = True) -> Tuple[str, float, float, float]:

        trial = trial.pick(channel)
        data = trial.data
        freq = trial.info['sfreq']
        if not average:
            return (channel, np.argmax(data) / freq, np.max(data))
        tmin_idx, tmax_idx = trial.time_as_index([tmin, tmax],
                                                 use_rounding=True)
        data = data[:, tmin_idx:tmax_idx]

        return (channel, trial.times[tmin_idx + np.argmax(data)], np.max(data),
                np.mean(data))

    def compute_peak(self, stim: str, thypothesis: float, offset: float,
                     channels: list[str]) -> pd.DataFrame:

        assert type(self.epochs) != np.ndarray, "Run compute_epochs first!"
        if isinstance(self.epochs, list):
            erp_dfs = []
            for epoch in self.epochs:
                erp_dfs.append(
                    self._get_erp_df(epoch, stim, thypothesis, offset,
                                     channels))
            return erp_dfs
        else:
            return self._get_erp_df(self.epochs, stim, thypothesis, offset,
                                    channels)

    def _get_erp_df(self, epochs: mne.Epochs, stim: str, thypothesis: float,
                    offset: float, channels: list[str]):
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
            channel, latency, peak, mamp = self.get_peak_channel(
                trial, channels[0], thypothesis - offset, thypothesis + offset)
            latency = int(round(latency * 1e3))  # convert to milliseconds
            peak = int(round(peak * 1e6))  # convert to µV
            mamp = int(round(mamp * 1e6))  # convert to µV
            peak_values['channel'].append(channel.strip())
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
