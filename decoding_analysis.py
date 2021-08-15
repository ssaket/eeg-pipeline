from dataclasses import dataclass

from dataclasses import dataclass, field
import logging

import sklearn
from pipeline import P3_EVENTS_MAPINGS

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from mne.decoding import Vectorizer

from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from abc import ABC, abstractmethod
from typing import List, Tuple
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
class SKLearnPipelineDecoder(Classifier):
    pipeline: Pipeline
    X: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    epochs: mne.Epochs = field(init=False, repr=False)
    timeVec: np.ndarray = field(init=False, repr=False)
    t_scores: List[float] = field(init=False, default_factory=list, repr=False)

    def fit(self, epochs: mne.Epochs, labels: np.ndarray, resampling_freq: float = 40) -> None:
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
            if w_tmax > timeVec[len(timeVec)-1]:
                continue
            # Crop data into time-window of interest
            X = self.epochs.crop(w_tmin, w_tmax).get_data()

            # Save mean scores over folds for each frequency and time window
            self.t_scores.append(np.mean(cross_val_score(estimator=self.pipeline, X=X, y=self.y,
                                                         scoring='roc_auc', cv=2,
                                                         n_jobs=2), axis=0))
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
    data: np.ndarray = field(init=False, repr=False, default=np.ndarray((0)))
    labels: np.ndarray = field(init=False)
    classifier: Classifier = field(init=False)

    def get_train(self, channels: list[str], feature_transform: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        self.data = np.array(self.epochs.get_data(picks=channels).mean(axis=2))
        self.labels = np.array(self.epochs.events[:, -1])

        if feature_transform:
            return self.feature_transform(self.data), self.labels_transform()
        else:
            return self.data, self.labels

    def feature_transform(self, transformer: FeatureTransformer = MNECSPTransformer(2)) -> Tuple[np.ndarray, np.ndarray]:
        labels = self.labels_transform()
        data = transformer.transform(self.epochs.get_data(), labels)
        return data, labels

    def labels_transform(self, n_classes=2) -> np.ndarray:
        _, rare, _ = P3_EVENTS_MAPINGS()
        wanted_keys = [self.epochs.event_id[key]
                       for key in self.epochs.event_id if int(key.split('/')[-1]) in rare]
        rare_stims = np.array([self.epochs.events[key]
                              for key in wanted_keys])[:, -1]
        labels = np.where(np.isin(self.labels, rare_stims), 1, 2)
        return labels

    def train(self, data, labels, classifier: Classifier) -> None:
        assert self.data.size > 1, "data and labels are not present, Run get_train method first!"
        self.classifier = classifier
        classifier.fit(data, labels)

    def predict(self, data, labels, classifier: Classifier = None) -> float:
        assert self.data.size > 0, "data and labels are not present, Run get_train method first!"
        if classifier:
            score = classifier.predict(data, labels)
        else:
            score = self.classifier.predict(data, labels)
        return score
