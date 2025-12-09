import os
from abc import ABC, abstractmethod

class Base(ABC):
    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    def check_assets(self, path='assets/'):
        os.makedirs(path, exist_ok=True)
        return os.path.exists(path)