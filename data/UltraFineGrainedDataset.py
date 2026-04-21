import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from config import UltraRoot


class UltraFineGrainedDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        train=True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        task="ncd",
    ):
        self.dataset_name = dataset_name
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.task = task
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")
        else:
            print("Files already downloaded and verified")

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        if self.train:
            self.data = pd.read_csv(
                os.path.join(UltraRoot[self.dataset_name], "anno/train.txt"),
                sep=" ",
                names=["filepath", "target"],
            )
        else:
            self.data = pd.read_csv(
                os.path.join(UltraRoot[self.dataset_name], "anno/test.txt"),
                sep=" ",
                names=["filepath", "target"],
            )

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(UltraRoot[self.dataset_name], "images", row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(UltraRoot[self.dataset_name], "images", sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.task == "ncd":
            return img, target, self.uq_idxs[idx]
        elif self.task == "gcd":
            return img, target, self.uq_idxs[idx]


# class SoyAgeing(Dataset):
#     base_folder = {
#         'SoyAgeing-R1': 'SoyAgeing-R1/R1',
#         'SoyAgeing-R3': 'SoyAgeing-R3/R3',
#         'SoyAgeing-R4': 'SoyAgeing-R4/R4',
#         'SoyAgeing-R5': 'SoyAgeing-R5/R5',
#         'SoyAgeing-R6': 'SoyAgeing-R6/R6'
#     }
#
#     def __init__(self, train=True, transform=None, target_transform=None, loader=default_loader,
#                  subset="SoyAgeing-R1", task="ncd"):
#
#         self.train = train
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader
#         self.subset = subset
#         self.task = task
#         if not self._check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You can use download=True to download it')
#         else:
#             print('Files already downloaded and verified')
#
#         self.uq_idxs = np.array(range(len(self)))
#
#     def _load_metadata(self):
#         if self.train:
#             self.data = pd.read_csv(os.path.join(UltraRoot['SoyAgeing'], self.base_folder[self.subset], 'anno/train.txt'),
#                                     sep=' ', names=['filepath', 'target'])
#         else:
#             self.data = pd.read_csv(os.path.join(UltraRoot['SoyAgeing'], self.base_folder[self.subset], 'anno/test.txt'),
#                                     sep=' ', names=['filepath', 'target'])
#
#     def _check_integrity(self):
#         try:
#             self._load_metadata()
#         except Exception:
#             return False
#
#         for index, row in self.data.iterrows():
#             filepath = os.path.join(self.root, self.base_folder[self.subset], 'images', row.filepath)
#             if not os.path.isfile(filepath):
#                 print(filepath)
#                 return False
#         return True
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         sample = self.data.iloc[idx]
#         path = os.path.join(self.root, self.base_folder[self.subset], 'images', sample.filepath)
#         target = sample.target - 1  # Targets start at 1 by default, so shift to 0
#         img = self.loader(path)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         if self.task == "ncd":
#             return img, target
#         elif self.task == "gcd":
#             return img, target, self.uq_idxs[idx]
