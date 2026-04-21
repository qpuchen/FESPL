import torch
import torchvision.transforms as T

from PIL import ImageFilter, ImageOps
import random


class DiscoverTargetTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, y):
        y = self.mapping[y]
        return y


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        v = torch.rand(1) * 256
        return ImageOps.solarize(img, v)


class Equalize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        return ImageOps.equalize(img)


def get_multicrop_transform(dataset, mean, std):
    if dataset == "ImageNet":
        return T.Compose(
            [
                T.RandomResizedCrop(size=96, scale=(0.08, 0.5)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    elif "cifar" in dataset:
        return T.Compose(
            [
                T.RandomResizedCrop(size=18, scale=(0.3, 0.8)),
                T.RandomHorizontalFlip(),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                Solarize(p=0.2),
                Equalize(p=0.2),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )


class MultiTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        return [t(x) for t in self.transforms]


def get_transforms(dataset="imagenet", args=None):
    multicrop, num_large_crops, num_small_crops = (
        args.multicrop,
        args.num_large_crops,
        args.num_small_crops,
    )
    if "imagenet" in dataset.lower():
        dataset = "ImageNet"
    mean, std = {
        "cifar10": [(0.491, 0.482, 0.447), (0.202, 0.199, 0.201)],
        "cifar100": [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)],
        "ImageNet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    }[dataset]
    # print("Remove RandomGrayscale and GaussianBlur to accelerate training!")

    transform = {
        "ImageNet": {
            "unsupervised": T.Compose(
                [
                    T.RandomResizedCrop(224, (0.5, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.5),
                    T.RandomGrayscale(p=0.2),
                    T.RandomApply([GaussianBlur([0.1, 2.0])], p=0.2),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "eval": T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
        },
        "cifar100": {
            "unsupervised": T.Compose(
                [
                    T.RandomChoice(
                        [
                            T.RandomCrop(32, padding=4),
                            T.RandomResizedCrop(32, (0.5, 1.0)),
                        ]
                    ),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                    Solarize(p=0.1),
                    Equalize(p=0.1),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "supervised": T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "eval": T.Compose(
                [
                    T.CenterCrop(32),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
        },
        "cifar10": {
            "unsupervised": T.Compose(
                [
                    T.RandomChoice(
                        [
                            T.RandomCrop(32, padding=4),
                            T.RandomResizedCrop(32, (0.5, 1.0)),
                        ]
                    ),
                    T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.6),
                    Solarize(p=0.1),
                    Equalize(p=0.1),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "supervised": T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
            "eval": T.Compose(
                [
                    T.CenterCrop(32),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            ),
        },
    }[dataset]

    train_transform = transform["unsupervised"]
    train_transforms = [train_transform] * num_large_crops
    if multicrop:
        multicrop_transform = get_multicrop_transform(dataset, mean, std)
        train_transforms += [multicrop_transform] * num_small_crops
    train_transform = MultiTransform(train_transforms)

    test_transform = transform["eval"]

    return (train_transform, test_transform)
