from FineGrained.data_utils import MergedDataset

from FineGrained.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from FineGrained.stanford_cars import get_scars_datasets
from FineGrained.cub import get_cub_datasets
from FineGrained.fgvc_aircraft import get_aircraft_datasets

from copy import deepcopy
import pickle
import os

from config import osr_split_dir


get_dataset_funcs = {
    "cifar10": get_cifar_10_datasets,
    "cifar100": get_cifar_100_datasets,
    "cub": get_cub_datasets,
    "aircraft": get_aircraft_datasets,
    "scars": get_scars_datasets,
}


def get_datasets(dataset, train_transform, test_transform, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """
    dataset = dataset.lower()
    #
    if dataset.lower() not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset]
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
    if args.pretrain:
        train_dataset = deepcopy(datasets["train_labelled"])
        # train_dataset = deepcopy(datasets['train_unlabelled'])
    else:
        train_dataset = MergedDataset(
            labelled_dataset=deepcopy(datasets["train_labelled"]),
            unlabelled_dataset=deepcopy(datasets["train_unlabelled"]),
        )

    test_dataset = datasets["test"]
    test_seen_dataset = datasets["test_seen"]
    unlabelled_train_examples_test = deepcopy(datasets["train_unlabelled"])
    unlabelled_train_examples_test.transform = test_transform
    # unlabelled_train_examples_test = datasets["val"]
    return (
        train_dataset,
        test_dataset,
        unlabelled_train_examples_test,
        test_seen_dataset,
    )


def get_class_splits(args):
    # For FGVC datasets, optionally return bespoke splits
    if args.dataset in ("scars", "cub", "aircraft"):
        if hasattr(args, "use_ssb_splits"):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset == "cifar10":
        args.image_size = 32
        args.train_classes = range(5)
        args.unlabeled_classes = range(5, 10)

    elif args.dataset == "cifar100":
        args.image_size = 224
        if args.num_labeled_classes == 50:
            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)
        else:
            args.train_classes = range(80)
            args.unlabeled_classes = range(80, 100)

    elif args.dataset == "herbarium_19":
        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, "herbarium_19_class_splits.pkl")

        with open(herb_path_splits, "rb") as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits["Old"]
        args.unlabeled_classes = class_splits["New"]

    elif args.dataset == "imagenet_100":
        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

    elif args.dataset == "imagenet1k":
        args.image_size = 224
        args.train_classes = range(882)
        args.unlabeled_classes = range(882, 882 + 30)
    elif args.dataset == "TinyImagenet":
        args.image_size = 224
        args.train_classes = range(100)
        args.unlabeled_classes = range(100, 200)

    elif args.dataset == "oxfordiiitpet":
        args.image_size = 224
        args.train_classes = range(19)
        args.unlabeled_classes = range(19, 37)

    elif args.dataset == "scars":
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

    elif args.dataset == "aircraft":
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

    elif args.dataset == "cub":
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

    else:
        raise NotImplementedError
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    if args.unknown_cluster:
        args.num_unlabeled_classes = round(args.num_unlabeled_classes * (1 + args.cluster_error_rate))
        # if args.dataset == "cub":
        #     args.num_unlabeled_classes = 63
        # elif args.dataset == "aircraft":
        #     args.num_unlabeled_classes = 31
        # elif args.dataset == "scars":
        #     args.num_unlabeled_classes = 77
        # else:
        #     raise NotImplementedError
    print("Number of unlabeled classes: {}".format(args.num_unlabeled_classes))
    return args
