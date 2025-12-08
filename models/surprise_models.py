from surprise import AlgoBase
from .base_ import Base
import pickle

class SurpriseModel(Base):
    def __init__(self, model: AlgoBase, save=True):
        self.model = model
        self.is_fitted = False
        self.save = save

    def fit(self, trainset):
        print(f"Training Model: {self.model.__class__.__name__}...")
        self.model.fit(trainset)
        self.is_fitted = True
        return self

    def predict(self, user_id, item_id):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        prediction = self.model.predict(uid=user_id, iid=item_id)
        return prediction.est

    def test(self, data):
        return self.model.test(data)

    def save_model(self):
        try:
            with open(f'{self.model.__class__.__name__}.pkl', 'wb') as f:
                pickle.dumps(self.model)
        except:
            raise Exception('Something went wrong')

    def load_model(self, path):
        if path is None:
            path = f'{self.model.__class__.__name__}.pkl'

        elif not(f'{self.model.__class__.__name__}') in path:
            raise Exception('Couldnt load the model')

        with open(path, 'rb') as f:
            return pickle.load(f)