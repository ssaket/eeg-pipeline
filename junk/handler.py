from enum import Enum


class STEP(Enum):
    DISPLAY_DESCRIPTION = 1
    CLEANING_FILTERING = 2
    ERP_PEAK_ANALYSIS = 3
    MULTIVARIATE_ANYALYSIS = 4
    DECODING_ANALYSIS = 5


class Handler(object):
    
    def __init__(self):
        self.nextHandler = None

    def handle(self, request):
        self.nextHandler.handle(request)


class DisplayDescriptionHandler(Handler):
    step = STEP.DISPLAY_DESCRIPTION

    def handle(self, payload):
        if payload.step == STEP.CLEANING_FILTERING:
            self.start_process(payload.raw)
        else:
            super(DisplayDescriptionHandler, self).handle(payload)

    def display(self):
        return "Hello"


class PreProcessingHandler(Handler):
    step = STEP.CLEANING_FILTERING

    def handle(self, payload):
        if payload.step == STEP.CLEANING_FILTERING:
            self.start_process(payload.raw)
        else:
            super(PreProcessingHandler, self).handle(payload)

    def start_process(self):
        pass


class ERPPeakAnalysisHandler(Handler):
    step = STEP.ERP_PEAK_ANALYSIS

    def handle(self, request):
        if request.format_ == STEP.CLEANING_FILTERING:
            self.start_process(request.title, request.text)
        else:
            super(ERPPeakAnalysisHandler, self).handle(request)


class MultivariateAnalysisHandler(Handler):
    step = STEP.MULTIVARIATE_ANYALYSIS

    def handle(self, request):
        if request.format_ == STEP.CLEANING_FILTERING:
            self.start_process(request.title, request.text)
        else:
            super(MultivariateAnalysisHandler, self).handle(request)


class DecodingAnalysisHandler(Handler):
    step = STEP.MULTIVARIATE_ANYALYSIS

    def handle(self, request):
        if request.format_ == STEP.CLEANING_FILTERING:
            self.start_process(request.title, request.text)
        else:
            super(DecodingAnalysisHandler, self).handle(request)
