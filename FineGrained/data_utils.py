import numpy as np
from torch.utils.data import Dataset


def subsample_instances(dataset, prop_indices_to_subsample=0.8):
    np.random.seed(0)
    subsample_indices = np.random.choice(
        range(len(dataset)),
        replace=False,
        size=(int(prop_indices_to_subsample * len(dataset)),),
    )

    return subsample_indices


class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):
        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):
        # if np.random.rand() > 0.5:
        #     item = np.random.randint(len(self.labelled_dataset))
        #     img, label = self.labelled_dataset[item % len(self.labelled_dataset)]
        #     labeled_or_not = 1
        # else:
        #     item = np.random.randint(len(self.unlabelled_dataset))
        #     img, label = self.unlabelled_dataset[item % len(self.unlabelled_dataset)]
        #     labeled_or_not = 0

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1
        else:
            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0
        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.labelled_dataset) + len(self.unlabelled_dataset)
