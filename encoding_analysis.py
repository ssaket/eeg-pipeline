from dataclasses import dataclass
import json
import os
from pipeline import Pipeline
from mne_bids.path import BIDSPath
import pandas as pd
import seaborn as sns


@dataclass
class LinearModel():
    pass

def _get_stimulas(row):
    if row.value == 202 or row.value == 201: return None
    desc = row.description
    stim = desc.split('-')[1].split(',')[1].split(' ')[-1]
    return stim
    
def _get_condition(row):
    if row.value == 202 or row.value == 201: return None
    desc = row.description
    stim = desc.split('-')[1].split(',')[1].split(' ')[-1]
    target = desc.split('-')[1].split(',')[0].split(' ')[-1]
    if stim == target:
        return 'Rare'
    else:
        return 'Frequent'
    
if __name__ == '__main__':
    bids_root = os.path.join('data', 'P3')
    bids_path = BIDSPath(subject='001', session='P3', task='P3',
                         datatype='eeg', suffix='eeg', root=bids_root)
    evts_file = os.path.join(bids_path.root,'task-P3_events.json')
    events_desc = json.load(open(evts_file,))
    events_desc = events_desc['value']['Levels']
    

    pip = Pipeline(bids_path)
    pip.start_preprocessing()
    
    df = pip.get_events_df()
    df['description'] = df.apply(lambda row: events_desc[str(row.value)], axis=1)
    df['stim'] = df.apply(lambda row: _get_stimulas(row), axis=1)
    df['cond'] = df.apply(lambda row: _get_condition(row), axis=1)
    
    raw = pip.raw
    raw_cz = raw.pick_channels(['Cz'])
    
    print(df)
    df.plot()
    sns.pairplot(df, hue='trial_type')
