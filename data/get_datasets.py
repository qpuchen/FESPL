from data.data_utils import MergedDataset

from data.stanford_cars import get_scars_datasets
from data.stanford_cars import subsample_classes as subsample_dataset_scars
from copy import deepcopy
import pickle
import os
import numpy as np

from config import osr_split_dir

sub_sample_class_funcs = {
    "scars": subsample_dataset_scars,
}

get_dataset_funcs = {
    "scars": get_scars_datasets,
}


def get_datasets(dataset_name, train_transform, test_transform, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """

    #
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets = get_dataset_f(
        train_transform=train_transform,
        test_transform=test_transform,
        train_classes=args.train_classes,
        prop_train_labels=args.prop_train_labels,
        split_train_val=False,
    )

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(
        labelled_dataset=deepcopy(datasets["train_labelled"]),
        unlabelled_dataset=deepcopy(datasets["train_unlabelled"]),
    )

    test_dataset = datasets["test"]
    unlabelled_train_examples_test = deepcopy(datasets["train_unlabelled"])
    unlabelled_train_examples_test.transform = test_transform

    if args.mode == "supervised":
        test_dataset_label = subsample_dataset_scars(deepcopy(test_dataset), include_classes=args.train_classes)
        return datasets["train_labelled"], test_dataset_label

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets


def get_class_splits(args):
    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ("scars", "cub", "aircraft"):
        if hasattr(args, "use_ssb_splits"):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == "cifar10":
        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset_name == "cifar100":
        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

    elif args.dataset_name == "tinyimagenet":
        args.image_size = 64
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == "herbarium_19":
        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, "herbarium_19_class_splits.pkl")

        with open(herb_path_splits, "rb") as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits["Old"]
        args.unlabeled_classes = class_splits["New"]

    elif args.dataset_name == "imagenet_100":
        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == "scars":
        args.image_size = 224

        if use_ssb_splits:
            split_path = os.path.join(osr_split_dir, "scars_osr_splits.pkl")
            with open(split_path, "rb") as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info["known_classes"]
            open_set_classes = class_info["unknown_classes"]
            args.unlabeled_classes = open_set_classes["Hard"] + open_set_classes["Medium"] + open_set_classes["Easy"]

        else:
            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

    elif args.dataset_name == "aircraft":
        args.image_size = 224
        if use_ssb_splits:
            split_path = os.path.join(osr_split_dir, "aircraft_osr_splits.pkl")
            with open(split_path, "rb") as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info["known_classes"]
            open_set_classes = class_info["unknown_classes"]
            args.unlabeled_classes = open_set_classes["Hard"] + open_set_classes["Medium"] + open_set_classes["Easy"]

        else:
            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

    elif args.dataset_name == "cub":
        args.image_size = 224

        if use_ssb_splits:
            split_path = os.path.join(osr_split_dir, "cub_osr_splits.pkl")
            with open(split_path, "rb") as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info["known_classes"]
            open_set_classes = class_info["unknown_classes"]
            args.unlabeled_classes = open_set_classes["Hard"] + open_set_classes["Medium"] + open_set_classes["Easy"]

        else:
            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

    elif args.dataset_name == "chinese_traffic_signs":
        args.image_size = 224
        args.train_classes = range(28)
        args.unlabeled_classes = range(28, 56)

    else:
        args.image_size = 448
        args.train_classes = range(99)
        args.unlabeled_classes = range(99, 198)
        args.old_classes = args.train_classes
        args.new_classes = args.unlabeled_classes

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
