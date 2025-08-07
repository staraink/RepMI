import numpy as np
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample
from utils.utils import *
import torch
from mne.filter import resample
from scipy.spatial.distance import cdist
import pandas as pd
from collections import Counter
from utils.channel_list import *

class EEGDataset(Dataset):
    def __init__(self, args=None):
        self.dataset_name = args.dataset_name
        self.args = args

        if self.dataset_name == 'BNCI2014001-4':
            X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
            y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
        else:
            X = np.load('./data/' + self.dataset_name + '/X.npy')
            y = np.load('./data/' + self.dataset_name + '/labels.npy')
        print("original data shape:", X.shape, "labels shape:", y.shape)
        if self.dataset_name == 'BNCI2014004':
            self.paradigm = 'MI'
            self.num_subjects = len(self.args.sub)
            self.sample_rate = 250
            
            # Define trial indices for each subject
            subject_indices = {
                0: np.arange(160) + 400,
                1: np.arange(120) + 1120,
                2: np.arange(160) + 1800,
                3: np.arange(160) + 2540,
                4: np.arange(160) + 3280,
                5: np.arange(160) + 4000,
                6: np.arange(160) + 4720,
                7: np.arange(160) + 5480,
                8: np.arange(160) + 6200
            }
            
            # Collect indices for selected subjects
            indices = []
            for subject_id in self.args.sub:
                indices.append(subject_indices[subject_id])
            
            # Combine and select data
            indices = np.concatenate(indices, axis=0)
            X = X[indices]
            y = y[indices]
            
            # Resample data (keeping first 1000 time points)
            X_resampled = X[:, :, :1000]
            self.sample_rate = 250  # Note: This is redundant as it was set above
            X = X_resampled
        else:
            self.paradigm = None
            self.num_subjects = None
            self.sample_rate = None
            self.ch_num = None

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        print("preprocessed data shape:", X.shape, "preprocessed labels shape:", y.shape)

        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)  # Ensure label is of type long
