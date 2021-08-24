from dataclasses import dataclass, field
import mne
import pandas as pd
import numpy as np
from sklearn import linear_model
from mne.stats import linear_regression
from abc import ABC, abstractmethod

@dataclass
class Encoder(ABC):
    @abstractmethod
    def fit():
        pass

@dataclass
class P3_LinearModel(Encoder):
    model: linear_model.LinearRegression = linear_model.LinearRegression()
    betas: list = field(init=False, repr=False, default_factory=list)

    def fit(self, epochs: mne.Epochs):
        epochs.equalize_event_counts(['stimulus', 'response']); # because we have one extra response
        reaction_time = epochs['stimulus'].events[:,0] - epochs['response'].events[:,0]
        epochs = epochs['stimulus']
        df = self.get_encoding_data(epochs, 'stimulus', channels=['Pz'])

        X = np.zeros([epochs.events.shape[0], 6]) # 6 because we have 5 betas
        X[:,0] = 1 # intercept
        X[np.where(df['condition'] =="rare"), 5] = 1
        X[np.where(df['stimulus']=="B"),1] = 1
        X[np.where(df['stimulus']=="C"),2] = 1
        X[np.where(df['stimulus']=="D"),3] = 1
        X[np.where(df['stimulus']=="E"),4] = 1

        X = np.hstack((X, reaction_time.reshape(-1, 1)))
            
        predictors = ['intercept', 'stim_b', 'stim_c', 'stim_d', 'stim_e', 'condition', 'reaction_time']
        df_mne = pd.DataFrame(X, columns=predictors)
        res = linear_regression(epochs['stimulus'], df_mne, names=predictors)
        self.betas.append(res)
    
    @staticmethod
    def get_encoding_data(epochs: mne.Epochs, condition: str, channels: list[str]):
        #reverse mapping of event_id from epochs
        inv_map = {v: k for k, v in epochs[condition].event_id.items()}
        event_names = [
            (i, inv_map[i]) for i in epochs[condition].events[:, -1]
        ]
        df = dict(epochs=[], stimulus=[], condition=[], code=[])
        for i, item in enumerate(event_names):
            code, name = item
            df['epochs'].append(i)
            df['stimulus'].append(name.split('/')[1])
            df['condition'].append(name.split('/')[2])
            df['code'].append(code)
        for channel in channels:
            df[channel] = epochs[condition].get_data(picks=[channel]).mean(axis=2).reshape(-1)
        return pd.DataFrame.from_dict(df)