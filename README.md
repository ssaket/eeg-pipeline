# EEG Pipeline

An EEG pipeline for encoding and decoding analysis built using MNE python. The pipeline design is highly influenced by [`sklearn.pipeline.Pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn-pipeline-pipeline). The steps are passed as a list.

In this repository, we use the pipeline to perform cleaning, ICA, ERP peak calculation, Mass Univariate analysis and Decoding Analysis on subjects from [ERPcore](https://erpinfo.org/erp-core) [1]() for the P300 task.

## Getting started

Quick Example

```python
bids_root = os.path.join('data', 'P3')
bids_path = BIDSPath(subject='001',
                            session='P3',
                            task='P3',
                            datatype='eeg',
                            suffix='eeg',
                            root=bids_root)
pipeline = Pipeline(self.bids_path, verbose=logging.ERROR)
pipeline.load_data()
pipeline.make_pipeline([
        SimpleMNEFilter(0.1, 50, 'firwin'),
        CleaningData(self.bids_path),
        PrecomputedICA(self.bids_path)
])
```

[![hello](https://img.shields.io/static/v1?label=Documentation&message=dev&color=yellowgreen)](https://ssaket.github.io/eeg-pipeline/)

## Setup

Use the environment.yml file to create a virtual env and install [mne-bids](https://mne.tools/mne-bids/stable/install.html)

## Features

- Provides all the tools from MNE python.
- Use Multiprocessing for processing multiple subjects.
- Takes list of processing steps and applies them sequentially.

## Classes

- `preprocessing.py`: It contains classes for preprocessing steps. You can create your own preprocessing class by extending the base class for than step.
- `erpanalysis.py`: It is a class containing functions for performing ERP analysis.
- `encoding_analysis.py`: It is a class containing functions for performing encoding analysis.
- `decoding_analysis.py`: It is a class containing functions for performing decoding analysis.
- `pipeline.py`: It is a class responsible for performing analysis steps. It further contains helper functions and classes.
- `test_pipeline.py`: It is a test class for `pipeline.py` and can be used for testing processing steps as well.

## Notebooks

Notebooks below have been move to repo: [eeg-analysis-notebooks](https://github.com/ssaket/eeg-analysis-notebooks)

- `preprocessing.ipynb`: We discuss how to perform preprocessing steps like applying filters, removing bad channels, annotating bad segments from EEG data. We also discuss on performing ICA like picking up and removing bad ICA components.

- `erpanalysis.ipynb`: we disscuss on peak to peak amplitude, diffrence between conditions and stimulus/

- `encoding.ipynb`: we discuss and analyse how to fit a mass univariate model on EEG data.

- `decoding.ipynb`: we disscuss and analyse classification scores based on Sensor-space decoding: decoding over time. We also discuss decoding in sensor space data.

## Structure

```shell
+-- cleaning
|   +-- badChannels
        +-- ..files containing bad channels for 3-subjects
|   +-- badSegments
        +-- ..files containing bad segments for 3-subjects
+-- data
|   +-- sub_1
|   +-- ..
+-- docs
|   +-- ..
+-- images
|   +-- ...
|   +-- plots of discussions
+-- src
```

## TODOs
[ ] Replace list comprehension with generators
[ ] Add Decoder based on Deep-learing
[ ] Do performance testing 

## References

[1] Emily S. Kappenman, Jaclyn L. Farrens, Wendy Zhang, Andrew X. Stewart, Steven J. Luck,
ERP CORE: An open resource for human event-related potential research,
NeuroImage,
Volume 225,
2021,
117465,
ISSN 1053-8119

[2] EEG Preprocessing and analysis course, <https://github.com/s-ccs/course_eeg_SS2021>
