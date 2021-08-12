from handler import Handler, STEP
from mne_bids import (BIDSPath, read_raw_bids, print_dir_tree)
import mne



class Pipeline:

    def __init__(self, steps=[STEP.DISPLAY_DESCRIPTION]):
        # assert isinstance(bids_path, BIDSPath)
        assert all(isinstance(step, STEP) for step in steps)

        self.steps: list[STEP] = steps
        self.handlers: list[Handler] = self._handlers(steps)

        # self.events, self.event_ids = mne.events_from_annotations(self.raw)

    def _handlers(self, steps):
        subclasses = Handler.__subclasses__()
        handlers = [handler() for _, handler in zip(
            steps, subclasses) if _ == handler.step]

        for idx in range(len(handlers) - 1):
            handlers[idx].nextHandler = handlers[idx+1]
        return handlers

    def add(self, step):
        assert isinstance(step, STEP)
        self.steps.append(step)
        self.handlers = self._handlers(self.steps)

    def process(self, raw):
        assert isinstance(raw, mne.io.Raw)

        for handler in self.handlers:
            payload = {'raw': raw, 'step': handler.step}
            handler.handle(payload)


if __name__ == '__main__':
    pipe = Pipeline('')
