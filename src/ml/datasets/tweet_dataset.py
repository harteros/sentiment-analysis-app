from pandas import DataFrame
from torch.utils.data import Dataset


class TweetDataset(Dataset):
    def __init__(self, data: DataFrame):
        self.data = data
        self.tweets = list(data.text)
        self.labels = list(data.target)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        return {'tweet': self.tweets[index], 'label': self.labels[index]}
