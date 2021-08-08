from dataclasses import dataclass
from mne_bids import (BIDSPath, read_raw_bids)
import os, mne

from filter import Filter

@dataclass
class EEG_OBJ:
    path: str = ''
    sub: int = 0
    
    def set_montage(self) -> None:
        montage_dir = os.path.join(os.path.dirname(mne.__file__),
                           'channels', 'data', 'montages')
        print('\nBUILT-IN MONTAGE FILES')
        print('======================')
        print(sorted(os.listdir(montage_dir)))
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        self.raw.set_montage(ten_twenty_montage, match_case=False)
        
    
    def load_data(self) -> None:
        bids_root = os.path.join('data', 'P3')
        bids_path = BIDSPath(subject='001', session='P3', task='P3', datatype='eeg', suffix='eeg', root=bids_root)
        raw = read_raw_bids(bids_path=bids_path)
        events, event_id = mne.events_from_annotations(raw)
        self.raw = raw.load_data()
        self.events = events
        self.event_id = event_id;
        
        
        
if __name__=='__main__':
    obj = EEG_OBJ()
    obj.load_data()
    print(obj.event_id, obj.events)
    
    Filter.apply(obj)