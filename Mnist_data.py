import gzip
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class mnist_dataset(Dataset):

    def __init__(self, path, is_valid=False):
        with gzip.open(path/'mnist.pkl.gz', 'rb') as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
        self.x = torch.from_numpy(x_train)
        self.y = torch.from_numpy(y_train)
        self.len = x_train.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
