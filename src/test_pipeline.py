import logging
from encoding_analysis import P3_LinearModel
from erpanalysis import ERPAnalysis
from preprocessing import CleaningData, PrecomputedICA, SimpleMNEFilter
from decoding_analysis import EEGDecoder

from pipeline import MultiPipeline, Pipeline
from mne_bids import BIDSPath
import unittest
import os


class PipelineTestCase(unittest.TestCase):

    def setUp(self):
        bids_root = os.path.join('data', 'P3')
        self.bids_path = BIDSPath(subject='001',
                                  session='P3',
                                  task='P3',
                                  datatype='eeg',
                                  suffix='eeg',
                                  root=bids_root)
        self.pipeline = Pipeline(self.bids_path, verbose=logging.ERROR)
        self.erp = ERPAnalysis(-0.1, 1)
        self.pipeline.load_data()
        self.pipeline.set_montage()
        self.pipeline.set_custom_events_mapping(task='P3')

    def test_preprocessing(self):
        self.pipeline.make_pipeline([
            SimpleMNEFilter(0.1, 50, 'firwin'),
            CleaningData(self.bids_path),
            PrecomputedICA(self.bids_path)
        ])
        # self.assertEqual(self.widget.size(), (50, 50),
        #                  'incorrect default size')

    def test_erp_epochs(self):
        self.pipeline.make_pipeline(
            SimpleMNEFilter(0.1, 50, 'firwin'),
            [CleaningData(self.bids_path),
             PrecomputedICA(self.bids_path)])
        erp_clean = ERPAnalysis(-0.1, 0.8)
        erp_clean.compute_epochs(self.pipeline.raw,
                                 self.pipeline.events,
                                 self.pipeline.event_ids,
                                 baseline=(0, None),
                                 reject_by_annotation=False)
        print(erp_clean.epochs)

    def test_erp_peaks(self):
        self.pipeline.make_pipeline([
            SimpleMNEFilter(0.5, 50, 'firwin'),
            CleaningData(self.bids_path),
            PrecomputedICA(self.bids_path)
        ])
        erp_clean = ERPAnalysis(-0.1, 0.8)
        erp_clean.compute_epochs(self.pipeline.raw,
                                 self.pipeline.events,
                                 self.pipeline.event_ids,
                                 baseline=(0, None),
                                 reject_by_annotation=True)
        peak = erp_clean.compute_peak('freq', 0.3, 0.1, ['Pz', 'CPz'])
        erp_clean.get_encoding_data()
        print(peak)

    def test_all_subject(self):
        self.pipeline.load_multiple_subjects(40)
        print("done")


class MultiPipelineTestCase(unittest.TestCase):

    def setUp(self):
        bids_root = os.path.join('data', 'P3')
        self.multi_pipeline = MultiPipeline(bids_root)
        self.erp = ERPAnalysis(-0.1,
                               0.8,
                               baseline=(None, 0),
                               reject_by_annotation=True,
                               all_subjects=True)

    def test_erp_analysis(self):
        self.multi_pipeline.start_erp_analysis(self.erp)

    def test_encoding_analysis(self):
        self.test_erp_analysis()
        encoder = P3_LinearModel()
        self.multi_pipeline.start_encoding_analysis(self.erp, encoder)

    def test_erp_epochs(self):
        self.erp.compute_peak('rare', 0.3, 0.1, ['Pz', 'CPz'], 'pos')
        self.erp.compute_peak('freq', 0.3, 0.1, ['Cz'], 'pos')

    def decoding_across_time(self):
        self.multi_pipeline.start_decoding()

    def test(self):
        pass


class DecodingAnalysisTestCase(unittest.TestCase):

    def setUp(self):
        bids_root = os.path.join('data', 'P3')
        self.bids_path = BIDSPath(subject='030',
                                  session='P3',
                                  task='P3',
                                  datatype='eeg',
                                  suffix='eeg',
                                  root=bids_root)
        self.pipeline = Pipeline(self.bids_path, verbose=logging.ERROR)
        self.pipeline.load_data()
        self.pipeline.set_montage()
        self.pipeline.set_custom_events_mapping(task='P3')
        self.pipeline.make_pipeline([
            SimpleMNEFilter(0.1, 50, 'firwin'),
            CleaningData(self.bids_path),
            PrecomputedICA(self.bids_path)
        ])

    def test_feature_transform(self):
        decoder = EEGDecoder('stimulus', (-0.2, 1), (0.0, 0.8),
                             self.pipeline.raw)
        csp_data, labels = decoder.feature_transform()

    def test_get_train_from_channels(self):
        decoder = EEGDecoder('stimulus', (-0.2, 1), (0.0, 0.8),
                             self.pipeline.raw)
        data, labels = decoder.get_train(channels=['Cz', 'CPz'],
                                         equalize_labels_count=True)

    def test_sliding_window(self):
        decoder = EEGDecoder('stimulus', (-0.2, 1), (0.0, 0.8),
                             self.pipeline.raw,
                             baseline=(None, 0),
                             reject_by_annotation=True)
        times, scores = decoder.run_sliding_()
        print(times)

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
    # suite.addTest(PipelineTestCase('test_erp_peaks'))
    # suite.addTest(PipelineTestCase('test_all_subject'))
    # suite.addTest(DecodingAnalysisTestCase('test_feature_transform'))
    # suite.addTest(DecodingAnalysisTestCase('test_get_train_from_channels'))
    # suite.addTest(DecodingAnalysisTestCase('test_sliding_window'))
    # suite.addTest(DecodingAnalysisTestCase('test_classify_over_time'))
    # suite.addTest(DecodingAnalysisTestCase('test_classify_all_stim'))
    suite.addTest(MultiPipelineTestCase('test_erp_analysis'))
    # suite.addTest(MultiPipelineTestCase('test_erp_epochs'))
    # suite.addTest(MultiPipelineTestCase('decoding_across_time'))
    # suite.addTest(MultiPipelineTestCase('test_encoding_analysis'))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
