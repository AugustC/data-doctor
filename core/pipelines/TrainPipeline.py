import core.ml as ml
from datetime import date
import os


class TrainPipeline:
    """
    TrainPipeline is a class that encapsulates the training process of the ML models
    """

    def __init__(self):
        self.classifier = ml.Classifier()
        self.regression = ml.Regression()

    def run(self, filename):
        today = date.today().strftime("%Y-%m-%d")

        classifier_output = f"experiments/classifier-{today}.model"
        self.classifier.load_dataset(filename=filename,
                                     label='chronic_obstructive_pulmonary_disease',
                                     drop_col=os.getenv('DROP_COLUMNS','').split(','))

        self.classifier.train_model()
        self.classifier.save_model(classifier_output)
        self.classifier.save_model('models/classifier.model')

        regression_output = f"experiments/regression-{today}.model"
        self.regression.load_dataset(filename=filename,
                                     label='alanine_aminotransferase',
                                     drop_col=['chronic_obstructive_pulmonary_disease',
                                               'alanine_aminotransferase',
                                               'patient_id'])
        self.regression.train_model()
        self.regression.save_model(regression_output)
        self.regression.save_model('models/regression.model')
