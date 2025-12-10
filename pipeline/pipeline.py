import numpy as np
import pandas as pd
from surprise import accuracy

from config import cfg


class Pipeline:
    def __init__(self, name, params={}, genre=cfg.genre):
        self.params = params
        self.name = name
        self.genre = genre
        self.dp = self.__get_data()
        self.model = self.__get_model(name)
        self.status = "Initializing"

    def fit(self):
        self.status = "Training Model"
        if self.name == "nn":
            train, val, test = self.dp.split_data(tensor=True)
            train_loader, val_loader = self.dp.make_batches(train, val, self.params.get("nn_batches", cfg.nn_batches))
            self.model.fit(train_loader, val_loader)
            self.status = "Scoring Model"
            predictions = self.model.test(test.to("cuda")).flatten()
            true = test[:, 2].numpy()
            mse = np.mean((true - predictions) ** 2)
            rmse = np.sqrt(mse)
            print(f"Score of NN is: {rmse}")

        elif self.name in ["knn", "bsl"]:
            train, test = self.dp.surprise_split()
            self.model.fit(train)
            self.status = "Scoring Model"
            print(f"Score of {self.name} on test: ", accuracy.rmse(self.model.test(test), verbose=True))
        else:
            raise Exception("Name {} is not allowed".format(self.name))
        self.status = "Idle"

    def save_model(self):
        self.status = "Saving Model"
        self.model.save_model()
        self.status = "Idle"

    def load_model(self, path=None):
        self.status = "Loading Model"
        self.model = self.model.load_model(path=path)
        self.status = "Idle"

    def __import_data(self):
        self.status = "Importing Data"
        return pd.read_csv(self.params.get("USER_DATA_PATH", cfg.USER_DATA_PATH))

    def __get_data(self):
        self.status = "Processing Data"
        from preprocessing import DataProcessor

        dp = DataProcessor(genre=False)
        dp.preprocessing(self.__import_data())
        return dp

    def __get_model(self, name):
        self.status = "Initializing Model"
        if name == "bsl":
            from surprise import BaselineOnly

            from models import SurpriseModel

            try:
                bsl_options = {
                    "n_epochs": self.params.get("bsl_n_epochs", cfg.bsl_n_epochs),
                    "method": "sgd",
                    "learning_rate": self.params.get("bsl_lr", cfg.bsl_lr),
                    "seed": cfg.seed,
                }

            except Exception as e:
                raise Exception("are u high?? using {} with these parameters {}, {}".format(self.name, self.params, e))

            algo = BaselineOnly(bsl_options=bsl_options)

            return SurpriseModel(algo)

        elif name == "knn":
            from surprise import KNNBasic

            from models import SurpriseModel

            sim_options = {"name": "cosine", "user_based": True}
            knn = KNNBasic(sim_options=sim_options)

            return SurpriseModel(knn)

        elif name == "nn":
            from models import RecSys, TorchModel

            # sending to cuda for training
            try:
                recsys = RecSys(
                    total_users=len(self.dp.user_mapping),
                    total_titles=len(self.dp.title_mapping),
                    user_emb_dim=self.params.get("nn_user_emb", cfg.nn_user_emb),
                    title_emb_dim=self.params.get("nn_title_emb", cfg.nn_title_emb),
                ).to("cuda")
                return TorchModel(
                    model=recsys,
                    epochs=self.params.get("nn_epochs", cfg.nn_epochs),
                    lr=self.params.get("nn_lr", cfg.nn_lr),
                    patience=self.params.get("nn_patience", cfg.nn_patience),
                )
            except Exception as e:
                raise Exception("are u high?? using {} with these parameters {}, {}".format(self.name, self.params, e))

        else:
            raise Exception("Name {} is not allowed".format(self.name))
