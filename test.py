import logging
from erpanalysis import ERPAnalysis
from preprocessing import CleaningData, PrecomputedICA, SimpleMNEFilter
from decoding_analysis import EEGDecoder, LDADecoder

from pipeline import Pipeline
from mne_bids import BIDSPath
import unittest
import os


class PipelineTestCase(unittest.TestCase):
    def setUp(self):
        bids_root = os.path.join('data', 'P3')
        self.bids_path = BIDSPath(subject='030', session='P3', task='P3',datatype='eeg', suffix='eeg', root=bids_root)
        self.pipeline = Pipeline(self.bids_path, verbose=logging.ERROR)
        self.erp =  ERPAnalysis(-0.1, 1)
        self.pipeline.load_data()
        self.pipeline.set_montage()

    def test_preprocessing(self):
        self.pipeline.make_pipeline([CleaningData(self.bids_path), SimpleMNEFilter(0.1, 50, 'firwin'), PrecomputedICA(self.bids_path)])
        # self.assertEqual(self.widget.size(), (50, 50),
        #                  'incorrect default size')

    def test_erp_analysis(self):
        self.pipeline.make_pipeline([CleaningData(self.bids_path), SimpleMNEFilter(0.1, 50, 'firwin'), PrecomputedICA(self.bids_path)])
        self.pipeline.compute_epochs(self.erp)
        
    def test_all_subject(self):
        self.pipeline.load_multiple_subjects(40)
        print("done")
    
        
        
class DecodingAnalysis(unittest.TestCase):
    def setUp(self):
        bids_root = os.path.join('data', 'P3')
        self.bids_path = BIDSPath(subject='030', session='P3', task='P3',datatype='eeg', suffix='eeg', root=bids_root)
        self.pipeline = Pipeline(self.bids_path, verbose=logging.ERROR)
        self.pipeline.load_data()
        self.pipeline.set_montage()
        self.pipeline.set_custom_events_mapping(task='P3')
        self.pipeline.make_pipeline([CleaningData(self.bids_path), SimpleMNEFilter(0.1, 50, 'firwin'), PrecomputedICA(self.bids_path)])
        
    def test_feature_transform(self):
        decoder = EEGDecoder('stimulus', (-0.2, 1), (0.0, 0.8), self.pipeline.raw)
        data, labels = decoder.get_train(channels=['Cz', 'CPz'])
        csp_data, labels = decoder.feature_transform()
        # decoder.train(csp_data, labels, LDADecoder())
        # decoder.predict(csp_data, labels)
        
    def test_classify_over_time(self):
        epoch_train = self.erp.epochs['stimulus']
        epoch_train = epoch_train.load_data().copy()
        
        decoder = EEGDecoder(epoch_train)
        _, labels = decoder.get_train(channels=['Cz', 'CPz'])
        labels = decoder.labels_transform()
        
        clf_svm = make_pipeline(Vectorizer(), StandardScaler(), svm.SVC(kernel='linear', C=1));
        clrs = SKLearnPipelineDecoder(clf_svm)
        clrs.fit(epoch_train, labels)
        scores = clrs.predict(w_size=1)
        
    def test_classify_all_stim(self):

        self.pipline.load_multiple_subjects(40)
        self.pipline.compute_epochs(self.erp)
        epoch_train = self.erp.epochs['stimulus']
        decoder = EEGDecoder(epoch_train)
        stim_data = decoder.get_all_stim()
        print("done")
                
def suite():
    suite = unittest.TestSuite()
    # suite.addTest(PipelineTestCase('test_preprocessing'))
    # suite.addTest(PipelineTestCase('test_erp_analysis'))
    # suite.addTest(PipelineTestCase('test_all_subject'))
    suite.addTest(DecodingAnalysis('test_feature_transform'))
    # suite.addTest(DecodingAnalysis('test_classify_over_time'))
    # suite.addTest(DecodingAnalysis('test_classify_all_stim'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
