from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.data_augment import get_transform
from data.UltraFineGrainedDataset import UltraFineGrainedDataset

dataset_splits = {
    "gcd": {
        "SoyAgeing-R1": [99, 198],
        "SoyAgeing-R3": [99, 198],
        "SoyAgeing-R4": [99, 198],
        "SoyAgeing-R5": [99, 198],
        "SoyAgeing-R6": [99, 198],
        "scars": [98, 196],
    },
    "ncd": {
        "SoyAgeing-R1": [99, 99],
        "SoyAgeing-R3": [99, 99],
        "SoyAgeing-R4": [99, 99],
        "SoyAgeing-R5": [99, 99],
        "SoyAgeing-R6": [99, 99],
        "SoyGene": [555, 555],
        "SoyGlobal": [969, 969],
        "SoyLobal": [100, 100],
        "Cotton": [40, 40],
    },
}


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def init_class_args(args):
    """
    :param args:
    :return: args w/ class num
    """

    if args.dataset not in dataset_splits[args.task].keys():
        import FineGrained

        args = FineGrained.get_datasets.get_class_splits(args)
        # raise ValueError('Undefined Ultra-FGVC dataset')
        old_class_num = args.num_labeled_classes
        new_class_num = args.num_unlabeled_classes
        args.old_classes = range(old_class_num)
        args.new_classes = range(old_class_num, old_class_num + new_class_num)
    else:
        old_class_num = dataset_splits[args.task][args.dataset][0]
        new_class_num = dataset_splits[args.task][args.dataset][1]

        if args.task == "ncd":
            args.old_classes = range(old_class_num)
            args.new_classes = range(old_class_num, old_class_num + new_class_num)
        elif args.task == "gcd":
            args.old_classes = range(old_class_num)
            args.new_classes = range(new_class_num)

    return args


def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype("bool")
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes):
    include_classes_ultra = np.array(include_classes) + 1
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r["target"]) in include_classes_ultra]

    # TODO: For now have no target transform
    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def subsample_instances(dataset, prop_index_to_subsample=0.5):
    np.random.seed(0)
    subsample_indices = np.random.choice(
        range(len(dataset)),
        replace=False,
        size=(int(prop_index_to_subsample * len(dataset)),),
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
        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:
            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0

        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)


def get_dataset_gcd(dataset_name, args, mode):
    train_transform = get_transform(
        image_size=args.image_size,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        task="gcd",
        mode="train",
        trans_type=args.trans_type,
    )
    test_transform = get_transform(
        image_size=args.image_size,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        task="gcd",
        mode="test",
        trans_type=args.trans_type,
    )
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # Get original dataset splits
    np.random.seed(0)

    train_set_whole = UltraFineGrainedDataset(
        dataset_name=dataset_name, train=True, transform=train_transform, task=args.task
    )
    train_set_label = subsample_classes(deepcopy(train_set_whole), include_classes=args.old_classes)

    # Subset train_set_label with prop_train_labels (default=0.5)
    subsample_indexes = subsample_instances(train_set_label, prop_index_to_subsample=args.prop_train_labels)
    train_set_label = subsample_dataset(train_set_label, subsample_indexes)

    #
    unlabel_index = set(train_set_whole.uq_idxs) - set(train_set_label.uq_idxs)
    train_set_unlabel = subsample_dataset(deepcopy(train_set_whole), np.array(list(unlabel_index)))

    test_set_whole = UltraFineGrainedDataset(
        dataset_name=dataset_name,
        train=False,
        transform=test_transform,
        task=args.task,
    )
    test_set_label = subsample_classes(deepcopy(test_set_whole), include_classes=args.old_classes)
    if mode == "supervised":
        return train_set_label, test_set_label

    train_set = MergedDataset(
        labelled_dataset=deepcopy(train_set_label),
        unlabelled_dataset=deepcopy(train_set_unlabel),
    )
    train_set_unlabel_test = deepcopy(train_set_unlabel)
    train_set_unlabel_test.transform = test_transform

    return train_set, test_set_whole, train_set_unlabel_test


def get_dataloader_gcd(dataset_name, args, mode):
    if mode == "supervised":
        train_set_label, test_set_label = get_dataset_gcd(dataset_name, args, mode)
        train_loader_label = DataLoader(
            train_set_label,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader_label = DataLoader(
            test_set_label,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )
        return train_loader_label, test_loader_label

    train_set, test_set, train_set_unlabel_test = get_dataset_gcd(dataset_name, args, mode)
    len_label = len(train_set.labelled_dataset)
    len_unlabel = len(train_set.unlabelled_dataset)
    sample_weights = [1 if i < len_label else len_label / len_unlabel for i in range(len(train_set))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_set))
    train_loader = DataLoader(
        train_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=True,
    )
    test_loader_train_unlabel = DataLoader(
        train_set_unlabel_test,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
    )
    return train_loader, test_loader, test_loader_train_unlabel


def get_dataset_ncd(dataset_name, args, mode="supervised"):
    train_transform = get_transform(
        image_size=args.image_size,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        task="ncd",
        mode="train",
        trans_type=args.trans_type,
    )
    test_transform = get_transform(
        image_size=args.image_size,
        crop_pct=args.crop_pct,
        interpolation=args.interpolation,
        task="ncd",
        mode="test",
        trans_type=args.trans_type,
    )
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    train_set_whole = UltraFineGrainedDataset(
        dataset_name=dataset_name, train=True, transform=train_transform, task=args.task
    )
    train_set_label = subsample_classes(deepcopy(train_set_whole), include_classes=args.old_classes)

    test_set_whole = UltraFineGrainedDataset(
        dataset_name=dataset_name, train=False, transform=test_transform, task=args.task
    )
    test_set_label = subsample_classes(deepcopy(test_set_whole), include_classes=args.old_classes)

    if mode == "supervised":
        return train_set_label, test_set_label
    elif mode == "unsupervised":
        train_set_unlabel = subsample_classes(deepcopy(train_set_whole), include_classes=args.new_classes)
        train_set_unlabel_test = deepcopy(train_set_unlabel)
        train_set_unlabel_test.transform = test_transform

        train_set = MergedDataset(
            labelled_dataset=deepcopy(train_set_label),
            unlabelled_dataset=deepcopy(train_set_unlabel),
        )

        test_set_unlabel = subsample_classes(deepcopy(test_set_whole), include_classes=args.new_classes)
        return (
            train_set,
            train_set_unlabel_test,
            test_set_label,
            test_set_unlabel,
            test_set_whole,
        )


def get_dataloader_ncd(dataset_name, args, mode="supervised"):
    if mode == "supervised":
        train_set_label, test_set_label = get_dataset_ncd(dataset_name, args, mode)
        train_loader_label = DataLoader(
            train_set_label,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader_label = DataLoader(
            test_set_label,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )
        return train_loader_label, test_loader_label
    elif mode == "unsupervised":
        (
            train_set,
            train_set_unlabel_test,
            test_set_label,
            test_set_unlabel,
            test_set_whole,
        ) = get_dataset_ncd(dataset_name, args, mode)
        train_loader = DataLoader(
            train_set,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader_unlabel_train = DataLoader(
            train_set_unlabel_test,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader_label = DataLoader(
            test_set_label,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader_unlabel = DataLoader(
            test_set_unlabel,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader_whole = DataLoader(
            test_set_whole,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
        )
        # test_loader_unlabel_train: 20 from train for test;
        # test_loader_label: 80 from test for test;
        # test_loader_unlabel: 20 from test for test;
        # test_loader_whole: 80 + 20 from test for test.
        return (
            train_loader,
            test_loader_unlabel_train,
            test_loader_label,
            test_loader_unlabel,
            test_loader_whole,
        )
