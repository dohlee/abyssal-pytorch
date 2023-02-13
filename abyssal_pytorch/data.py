import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

class MegaDataset(Dataset):
    def __init__(self, df):
        """df: pandas dataframe for MegaTrain/MegaValidation/MegaTest datasets.
        """
        self.df = df.to_records()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df[idx]

        return {
            'seq': r['wt_seq'],
            'pos': r['pos'],
            'aa_mut': r['aa_mut'],
            'label': torch.tensor([r['ddG_ML']]).float(),
        }

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('../data/mega.test.csv')
    test = MegaDataset(df)

    test_loader = DataLoader(test, batch_size=10, shuffle=True)
    for batch in test_loader:
        print(batch['seq'])
        print(batch['pos'])
        print(batch['aa_mut'])
        print(batch['label'])
        break