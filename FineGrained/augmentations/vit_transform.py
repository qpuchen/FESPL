from torchvision import transforms

import torch


def get_transforms(transform_type="imagenet", args=None):
    if transform_type == "imagenet":
        # mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        image_size = 224
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = 3
        crop_pct = 0.875

        train_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
            ]
        )
        train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    else:
        raise NotImplementedError

    return (train_transform, test_transform)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]
