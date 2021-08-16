from dataclasses import dataclass, field
import logging

from pipeline import P3_EVENTS_MAPINGS
from mne.decoding import Vectorizer

import sklearn
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import seaborn as sns
import numpy as np
import mne


@dataclass
class Classifier(ABC):

    @abstractmethod
    def fit():
        pass

    @abstractmethod
    def predict() -> np.ndarray:
        pass


@dataclass
class LDADecoder(Classifier):
    lda: LinearDiscriminantAnalysis = field(
        init=False, default_factory=LinearDiscriminantAnalysis)

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        data = data.reshape(data.shape[0], -1)
        self.lda.fit(data, labels)

    def predict(self, data: np.ndarray, labels: np.ndarray) -> float:
        score = self.lda.score(data, labels)
        logging.info("Score {}".format(score))
        return score


@dataclass
class CVPipleline(Classifier):
    cv_pip: sklearn.pipeline.Pipeline
    train_data: np.ndarray
    train_labels: np.ndarray
    test_data: np.ndarray
    test_labels: np.ndarray

    def fit(self, **kwargs):
        self.cv_pip.fit(self.train_data, self.train_labels, **kwargs)

    def predict(self, **kwargs) -> np.ndarray:
        self.predictions = self.cv_pip.predict(self.test_data, **kwargs)

    def evaluate(self, target_names: list[str] = ['Rare', 'Frequent']):
        report = classification_report(self.test_labels,
                                       self.predictions,
                                       target_names=target_names)
        print('Clasification Report:\n {}'.format(report))

        acc = accuracy_score(self.test_labels, self.predictions)
        print("Accuracy of model: {}".format(acc))

        precision, recall, fscore, support = precision_recall_fscore_support(
            self.test_labels, self.predictions_lda, average='macro')
        print('Precision: {0}, Recall: {1}, f1-score:{2}'.format(
            precision, recall, fscore))
        return acc, fscore


@dataclass
class SKLearnPipelineDecoder(Classifier):
    pipeline: sklearn.pipeline.Pipeline
    X: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    epochs: mne.Epochs = field(init=False, repr=False)
    timeVec: np.ndarray = field(init=False, repr=False)
    t_scores: List[float] = field(init=False, default_factory=list, repr=False)

    def fit(self,
            epochs: mne.Epochs,
            labels: np.ndarray,
            resampling_freq: float = 40) -> None:
        self.epochs = epochs.copy().resample(resampling_freq)
        self.X = self.epochs.get_data()
        self.y = labels
        self.timeVec = epochs.times

    def predict(self, w_size) -> np.array:
        timeVec = self.timeVec[::10]
        for t, w_time in enumerate(timeVec):
            w_tmin = w_time - w_size / 2.
            w_tmax = w_time + w_size / 2.

            # stop the program if the timewindow is outside of our epoch
            if w_tmin < timeVec[0]:
                continue
            if w_tmax > timeVec[len(timeVec) - 1]:
                continue
            # Crop data into time-window of interest
            X = self.epochs.crop(w_tmin, w_tmax).get_data()

            # Save mean scores over folds for each frequency and time window
            self.t_scores.append(
                np.mean(cross_val_score(estimator=self.pipeline,
                                        X=X,
                                        y=self.y,
                                        scoring='roc_auc',
                                        cv=2,
                                        n_jobs=2),
                        axis=0))
        return np.array(self.t_scores)


@dataclass
class FeatureTransformer(ABC):

    @abstractmethod
    def transform() -> np.ndarray:
        pass


@dataclass
class MNECSPTransformer(FeatureTransformer):
    n_components: int

    def transform(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        csp = mne.decoding.CSP(self.n_components)
        csp.fit_transform(data, labels)
        return csp.transform(data)


@dataclass
class Decoding():
    epochs: mne.Epochs
    classifier: Classifier = field(init=False)

    def get_train(
            self,
            channels: list[str] = None,
            transform_feature: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if channels:
            data = np.array(self.epochs.get_data(picks=channels).mean(axis=2))
        else:
            data = np.array(self.epochs.get_data().mean(axis=2))

        if transform_feature:
            return self.feature_transform(data), self.labels_transform()
        else:
            return data, self.labels_transform()

    def get_all_stim(self):

        epoch_A = self.epochs['stimulus/A'].copy()
        epoch_B = self.epochs['stimulus/B'].copy()
        epoch_C = self.epochs['stimulus/C'].copy()
        epoch_D = self.epochs['stimulus/D'].copy()
        epoch_E = self.epochs['stimulus/E'].copy()

        data = {
            'A': {
                'epoch': epoch_A,
                'data': epoch_A.get_data(),
                'labels': self.labels_transform(epoch_A)
            },
            'B': {
                'epoch': epoch_B,
                'data': epoch_B.get_data(),
                'labels': self.labels_transform(epoch_B)
            },
            'C': {
                'epoch': epoch_C,
                'data': epoch_C.get_data(),
                'labels': self.labels_transform(epoch_C)
            },
            'D': {
                'epoch': epoch_D,
                'data': epoch_D.get_data(),
                'labels': self.labels_transform(epoch_D)
            },
            'E': {
                'epoch': epoch_E,
                'data': epoch_E.get_data(),
                'labels': self.labels_transform(epoch_E)
            },
        }

        return data

    def feature_transform(
        self, transformer: FeatureTransformer = MNECSPTransformer(2)
    ) -> Tuple[np.ndarray, np.ndarray]:
        labels = self.labels_transform()
        data = transformer.transform(self.epochs.get_data(), labels)
        return data, labels

    def labels_transform(self,
                         epochs: mne.Epochs = None,
                         n_classes: int = 2) -> np.ndarray:
        _epochs = self.epochs if epochs is None else epochs
        _labels = self.labels if epochs is None else epochs.events[:, -1]
        _, rare, _ = P3_EVENTS_MAPINGS()
        wanted_keys = [
            _epochs.event_id[key]
            for key in _epochs.event_id
            if int(key.split('/')[-1]) in rare
        ]
        rare_stims = np.array([_epochs.events[key] for key in wanted_keys])
        rare_stims = rare_stims[:, -1]
        labels = np.where(np.isin(_labels, rare_stims), 1, 2)
        return labels

    def train(self, data: np.ndarray, labels: np.ndarray,
              classifier: Classifier) -> None:
        assert self.data.size > 1, "data and labels are not present, Run get_train method first!"
        if classifier:
            classifier.fit()
        else:
            self.classifier.fit(data, labels)

    def predict(self, data, labels, classifier: Classifier = None) -> float:
        assert self.data.size > 0, "data and labels are not present, Run get_train method first!"
        if classifier:
            score = classifier.predict()
        else:
            score = self.classifier.predict(data, labels)
        return score

    def plotMetrics(tasks, labels, evalMetric, metricName, ax):
        ax = sns.barplot(x=tasks, y=metricName, data=evalMetric, hue=labels)
        return ax
