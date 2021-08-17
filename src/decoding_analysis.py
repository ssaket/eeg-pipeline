from dataclasses import dataclass, field
import logging

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, make_pipeline

from abc import ABC, abstractmethod
from typing import List, Tuple
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
    cv_pip: Pipeline
    train_data: np.ndarray
    test_data: np.ndarray
    train_labels: np.ndarray
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
            self.test_labels, self.predictions, average='macro')
        print('Precision: {0}, Recall: {1}, f1-score:{2}'.format(
            precision, recall, fscore))
        return acc, fscore


@dataclass
class OvertimePipelineDecoder(Classifier):
    pipeline: Pipeline
    X: np.ndarray = field(init=False, repr=False)
    y: np.ndarray = field(init=False, repr=False)
    epochs: mne.Epochs = field(init=False, repr=False)
    timeVec: np.ndarray = field(init=False, repr=False)
    t_scores: List[float] = field(init=False, default_factory=list, repr=False)

    def fit(self,
            epochs: mne.Epochs,
            labels: np.ndarray,
            resampling_freq: float = 40) -> None:
        self.epochs = epochs.load_data().resample(resampling_freq)
        self.X = self.epochs.get_data()
        self.y = labels
        self.timeVec = epochs.times

    def predict(self, w_size: float) -> np.array:
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
class EEGDecoder():
    condition: str
    epoch_times: Tuple[float, float]
    decoding_times: Tuple[float, float]
    raw: mne.io.Raw = field(repr=False)
    epochs: mne.Epochs = field(init=False, repr=False)
    score: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        events, ids = mne.events_from_annotations(self.raw)
        epochs = mne.Epochs(self.raw, events, ids, self.epoch_times[0], self.epoch_times[1], None, reject_by_annotation=False)
        self.epochs = epochs[self.condition].load_data().crop(self.decoding_times[0], self.decoding_times[1]).copy()

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
        _epochs = epochs if epochs else self.epochs
        _labels = epochs.events[:, -1] if epochs else self.epochs.events[:, -1]
        _, rare, _ = P3.EVENTS_MAPINGS()
        wanted_keys = [
            _epochs.event_id[key]
            for key in _epochs.event_id
            if int(key.split('/')[-1]) in rare
        ]
        rare_stims = np.array([_epochs.events[key] for key in wanted_keys])
        rare_stims = rare_stims[:, -1]
        labels = np.where(np.isin(_labels, rare_stims), 1, 2)
        return labels

    def plotMetrics(tasks, labels, evalMetric, metricName, ax):
        ax = sns.barplot(x=tasks, y=metricName, data=evalMetric, hue=labels)
        return ax
    
    def run_svm_decoder(self):
        from mne.decoding.transformer import Vectorizer
        from sklearn.preprocessing import StandardScaler
        from sklearn import svm
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        data, labels = self.get_train(channels=['Cz', 'CPz'])
        clf_svm_pip = make_pipeline(Vectorizer(), StandardScaler(), svm.SVC(random_state=42))
        parameters = {'svc__kernel':['linear', 'rbf', 'sigmoid'], 'svc__C':[0.1, 1, 10]}
        gs_cv_svm = GridSearchCV(clf_svm_pip, parameters, scoring='accuracy', cv=StratifiedKFold(n_splits=5), return_train_score=True);
        gs_cv_svm.fit(data, labels)
        logging.info('Best Parameters: {}'.format(gs_cv_svm.best_params_))
        logging.info('Best Score: {}'.format(gs_cv_svm.best_score_))
        return gs_cv_svm.best_score_, gs_cv_svm.best_params_

class P3:
    @abstractmethod
    def EVENTS_MAPINGS() -> Tuple[dict, list[int], list[int]]:
        blocks = np.array(
            [list(range(10 * x + 1, 10 * x + 6)) for x in range(1, 6)])
        rare = np.array([x + i for i, x in enumerate(range(11, 56, 10))]).tolist()
        freq = np.setdiff1d(blocks.flatten(), rare).tolist()

        stimlus = ['A', 'B', 'C', 'D', 'E']

        evts_stim = [
            'stimulus/' + stimlus[i] + '/' + str(alph)
            for i, x in enumerate(blocks)
            for alph in x
        ]
        evts_id = dict((i + 3, evts_stim[i]) for i in range(0, len(evts_stim)))
        evts_id[1] = 'response/201'
        evts_id[2] = 'response/202'
        return evts_id, rare, freq
