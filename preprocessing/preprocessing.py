import pandas as pd     
import torch
from surprise import Dataset, Reader, accuracy
from rec_utils import split_the_data, convert_to_list
from torch.utils.data import TensorDataset, DataLoader
from surprise.model_selection import train_test_split as sur_tts

class DataProcessor:
    def __init__(self, genre=False):
        self.user_mapping = None
        self.title_mapping = None
        self.reverse_user = None
        self.reverse_title = None
        self.data = None
        self.genre = genre

    def preprocessing(self, user_data, course=None, courses=None):
        if self.genre and (type(course) == pd.DataFrame and type(courses) == pd.DataFrame):
            course.dropna(subset='skills', inplace=True)
            courses.dropna(subset='skills', inplace=True)

            course['skills'] = course['skills'].apply(lambda x: convert_to_list(x))
            courses['skills'] = courses['skills'].apply(lambda x: convert_to_list(x))

            skills = pd.concat([course.explode('skills'), courses.explode('skills')], ignore_index=True)
            skill_onehot = pd.get_dummies(skills).astype(int).groupby('title_index').max().reset_index()

            user_data = user_data.merge(skill_onehot, on='title_index', how='left').fillna(0)

            self.data = user_data
        else:
            self.data = user_data

        self.make_mapping()

        self.data['user_index'] = self.data['user_index'].map(self.user_mapping)
        self.data['title_index'] = self.data['title_index'].map(self.title_mapping)
        
    def make_mapping(self):
        self.user_mapping = {
            k: x for x, k in enumerate(self.data['user_index'].unique())
        }
        
        self.title_mapping = {
            k: x for x, k in enumerate(self.data['title_index'].unique())
        }

        self.reverse_user = {v: k for k, v in self.user_mapping.items()}
        self.reverse_title = {v: k for k, v in self.title_mapping.items()}

    def unmap(self, index):
        user_index, title_index = index
        return self.reverse_user[user_index], self.reverse_title[index]

    def apply_mapping(self, data):
        try:
            user_index, title_index = data
            return self.user_mapping[user_index], self.title_mapping[title_index]
        except:
            raise Exception('New user or title is added so re-train the model')

    def split_data(self, tensor=True):
        data, test = split_the_data(self.data[['title_index', 'user_index', 'rating']])
        train, val = split_the_data(self.data)

        X_train = train.drop('rating', axis=1)
        X_val = val.drop('rating', axis=1)
        
        y_train = train['rating']
        y_val = val['rating']

        if tensor:
            X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float)  
            X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float)    
            y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float)  
            y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float) 
            
        return (X_train, y_train), (X_val, y_val), test

    def surprise_split(self):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.data[['user_index', 'title_index', 'rating']], reader)
        train, test = sur_tts(data, test_size=0.2, shuffle=True)

        return train, test

    def make_batches(self, train , test, batch_size):
        tensor_Xtr, tensor_ytr = train
        tensor_Xva, tensor_yva = test
        train_dataset = TensorDataset(tensor_Xtr, tensor_ytr)
        val_dataset   = TensorDataset(tensor_Xva, tensor_yva)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, val_loader