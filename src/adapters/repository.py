from torchsystem.storage import Models, Criterions, Optimizers, Datasets
from src.adapters.classifier import Classifier

class Repository:
    
    def __init__(self, experiment: str, folder: str = 'data/weights'):
        self.models = Models(folder, experiment)
        self.criterions = Criterions(folder, experiment)
        self.optimizers = Optimizers(folder, experiment)
        self.datasets = Datasets()

    def store(self, classifier: Classifier):
        self.models.store(classifier.model)

    def restore(self, classifier: Classifier):
        self.models.restore(classifier.model)