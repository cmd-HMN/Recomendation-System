import torch
from torch import nn
from torch import optim
from .base_ import Base
from .early_stopping import EarlyStopping
import numpy as np

class TorchModel(Base):
    def __init__(self, model: nn.Module, epochs=1500, lr=0.001, patience=100):
        self.patience=patience
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-2, momentum=0.7)
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=False)
        
    def fit(self, train_loader, val_loader):
        train_losses = []
        val_losses = []
        
        for epoch in range(self.epochs):
            self.model.train()
            loss_ = 0
            
            for x_, y_ in train_loader:
                x_ = x_.to('cuda', non_blocking=True)
                y_ = y_.to('cuda', non_blocking=True)
                
                self.optimizer.zero_grad()
                outputs = self.model(x_)
                loss = torch.sqrt(self.criterion(outputs, y_.view(-1, 1)))
                loss.backward()
                self.optimizer.step()
            
                loss_ += loss.item()
            
            train_losses.append(loss_ / len(train_loader))
            
            # validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for x_, y_ in val_loader:  
                    x_ = x_.to('cuda', non_blocking=True)
                    y_ = y_.to('cuda', non_blocking=True)
                    val_output = self.model(x_)
                    val_loss += torch.sqrt(self.criterion(val_output, y_.view(-1, 1))).item()
            
            val_losses.append(val_loss / len(val_loader))

            print(f'Epoch {epoch+1} - Train Loss: {train_losses[-1]:.4f} - Val Loss: {val_losses[-1]:.4f}')
            
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break

        return train_losses, val_losses 

    def save_model(self):
        torch.save(self.model.state_dict(), f'assets/{self.model.__class__.__name__}')

    def load_model(self, path=None):
        if path is None:
            path = f'assets/{self.model.__class__.__name__}'
        elif not (f'{self.model.__class__.__name__}' in path):
            raise Exception('Something went wrong')

        self.model.load_state_dict(torch.load(path))

    def test(self, test):
        self.model.eval()
        with torch.no_grad():
            output = self.model(test)

        return output.detach().cpu().numpy()
        
    def predict(self, user_index, title_index):
        x_ = torch.tensor(np.array([user_index, title_index]))
        x_ = x_.unsqueeze(0).to('cuda')
        self.model.eval()
        with torch.no_grad():
                output = self.model(x_)
            
        return output