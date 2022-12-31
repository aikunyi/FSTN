from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd

# traffic data
class Dataset_Dhfm(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        load_data = np.load(root_path)
        data = load_data.transpose()
        if type == '1':
            mms = MinMaxScaler(feature_range=(0, 1))
            data = mms.fit_transform(data)
        if self.flag == 'train':
            begin = 0
            end = int(len(data)*0.7)
            self.trainData = data[begin:end]
        if self.flag == 'val':
            begin = int(len(data)*0.7)
            end = int(len(data)*0.9)
            self.valData = data[begin:end]
        if self.flag == 'test':
            begin = int(len(data)*0.9)
            end = len(data)
            self.testData = data[begin:end]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_end = end + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        return data, next_data

    def __len__(self):
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len

# ECG/COVID dataset
class Dataset_ECG(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'test', 'val']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        data = pd.read_csv(root_path)

        if type == '1':
            mms = MinMaxScaler(feature_range=(0, 1))
            data = mms.fit_transform(data)
        data = np.array(data)
        if self.flag == 'train':
            begin = 0
            end = int(len(data)*self.train_ratio)
            self.trainData = data[begin:end]
        if self.flag == 'val':
            begin = int(len(data)*self.train_ratio)
            end = int(len(data)*(self.val_ratio+self.train_ratio))
            self.valData = data[begin:end]
        if self.flag == 'test':
            begin = int(len(data)*(self.val_ratio+self.train_ratio))
            end = len(data)
            self.testData = data[begin:end]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[next_begin:next_end]
        elif self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[next_begin:next_end]
        else:
            data = self.testData[begin:end]
            next_data = self.testData[next_begin:next_end]
        return data, next_data

    def __len__(self):
        # minus the label length
        if self.flag == 'train':
            return len(self.trainData)-self.seq_len-self.pre_len
        elif self.flag == 'val':
            return len(self.valData)-self.seq_len-self.pre_len
        else:
            return len(self.testData)-self.seq_len-self.pre_len
