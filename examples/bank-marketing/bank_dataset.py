import random

import torch
import numpy as np
from torch.utils.data import Dataset


class BankDataset(Dataset):
    """
    This class loads the bank marketing dataset for processing.
    The loaded dataset is already down-sampled for better illustrating
    the training and testing performance.
    """

    def __init__(self, path):
        """
        Args
        :param path: the dataset file path
        """
        full_data_table = np.genfromtxt(path, delimiter=',')
        data = torch.from_numpy(full_data_table).float()
        self.samples = data[:, :-1]
        self.labels = data[:, -1]

        # compute features' max and min values
        min_v, _ = self.samples.min(dim=0)
        max_v, _ = self.samples.max(dim=0)
        self.feature_min = min_v
        self.feature_max = max_v

        # normalization
        threshold = 1e-3
        self.samples = (self.samples - self.feature_min + threshold) / (self.feature_max - self.feature_min + threshold)
        self.mean_attr = self.samples.mean(dim=0)
        
        # bank dataset is highly imbalanced, down-sample the negative samples
        balance_num = int(sum(self.labels) * 2)
        self.balance_labels = torch.zeros(balance_num)
        self.balance_samples = torch.zeros(balance_num, self.samples.shape[1])
        balance_index = 0
        negative_counter = 0
        idxes = list(range(len(self.labels)))
        random.shuffle(idxes)
        for i in idxes:
            if self.labels[i] == 0 and negative_counter < balance_num / 2:
                negative_counter += 1
                self.balance_labels[balance_index] = self.labels[i]
                self.balance_samples[balance_index] = self.samples[i]
                balance_index += 1
            elif self.labels[i] == 1:
                self.balance_labels[balance_index] = self.labels[i]
                self.balance_samples[balance_index] = self.samples[i]
                balance_index += 1
            if balance_index >= balance_num:
                break

        print("Len(samples):", len(self.balance_labels), "Positive labels sum:", self.balance_labels.sum().item())

    def __len__(self):
        return len(self.balance_samples)

    def __getitem__(self, index):
        return self.balance_samples[index], self.balance_labels[index]
