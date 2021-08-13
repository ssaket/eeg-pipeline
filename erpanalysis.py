from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import mne
import logging


@dataclass
class ERPAnalysis():
    tmin: float
    tmax: float
    baseline: Optional[Tuple[Optional[int], int]] = None
    reject_by_annotation: bool = False
    epochs: mne.Epochs = field(init=False, repr=False, default=np.ndarray(0))

    def compute_epochs(self, raw: mne.io.Raw, events: np.ndarray, events_id: Dict) -> None:
        """There is a bug in MNE, even if we pass the default baseline (None, 0), we get slightly different results."""
        if self.baseline:
            self.epochs = mne.Epochs(raw, events=events, event_id=events_id,
                                     tmin=self.tmin, tmax=self.tmax, baseline=self.baseline, reject_by_annotation=self.reject_by_annotation)
        else:
            self.epochs = mne.Epochs(raw, events=events, event_id=events_id,
                                     tmin=self.tmin, tmax=self.tmax, reject_by_annotation=self.reject_by_annotation)

    def get_peak_channel(self, trial: mne.Evoked, channel: str, tmin: float, tmax: float, average: bool = True) -> Tuple[str, float, float, float]:

        trial = trial.pick(channel).copy()
        data = trial.data
        freq = trial.info['sfreq']
        if not average:
            return (channel, np.argmax(data)/freq, np.max(data))
        tmin_idx, tmax_idx = trial.time_as_index(
            [tmin, tmax], use_rounding=True)
        data = data[:, tmin_idx:tmax_idx]

        return (channel, trial.times[tmin_idx + np.argmax(data)], np.max(data), np.mean(data))

    def compute_peak(self, stim: str, thypothesis: float, offset: float, channels: list[str]) -> pd.DataFrame:

        assert type(self.epochs) != np.ndarray, "Run compute_epochs first!"
        peak_values = {'channel': [], 'peak_amp': [],
                       'mean_amp': [], 'latency': [], 'trial': []}
        _epochs: mne.Epochs = self.epochs[stim] if stim else self.epochs

        for ix, trial in enumerate(_epochs.iter_evoked()):
            # channel, _latency, _value = trial.get_peak(ch_type='eeg', tmin=thypothesis - offset, tmax=thypothesis + offset,
            #                                          return_amplitude=True, mode='pos')
            # if len(channels) != 0 and not channel in channels:
            #     continue

            channel, latency, peak, mamp = self.get_peak_channel(
                trial, channels[0], thypothesis - offset, thypothesis + offset)

            # assert value == _value and latency == _latency

            latency = int(round(latency * 1e3))  # convert to milliseconds
            peak = int(round(peak * 1e6))      # convert to µV
            mamp = int(round(mamp * 1e6))      # convert to µV
            peak_values['channel'].append(channel.strip())
            peak_values['peak_amp'].append(peak)
            peak_values['mean_amp'].append(mamp)
            peak_values['latency'].append(latency)
            peak_values['trial'].append(ix)

            logging.info('Trial {}: peak of {} µV and mean of {} µV at {} ms in channel {}'
                         .format(ix, peak, mamp, latency, channel))
        df = pd.DataFrame.from_dict(peak_values)
        del peak_values
        return df
