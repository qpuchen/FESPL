from torchvision import transforms
import random
from PIL import ImageFilter

import torch
from torchvision import transforms as T
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import str_to_pil_interp

class MyTransform:
    def __init__(self, image_size):

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])

        self.common_aug = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            # T.Resize((args.resize_size, args.resize_size),interpolation=str_to_pil_interp('bicubic')),
            # T.RandomCrop((args.img_size, args.img_size)),
            T.RandomResizedCrop(image_size, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            T.ColorJitter(0.3),
            T.RandomRotation(15),
            T.RandomHorizontalFlip(),
        ])

        self.base = T.Compose([
            self.common_aug,
            self.to_tensor,
        ])

    def __call__(self, image):
        img = self.base(image)

        return img

def get_transform(image_size, crop_pct, interpolation, trans_type="imagenet", task="ncd", mode="train"):
    mean, std = {
        "cifar10": [(0.491, 0.482, 0.447), (0.202, 0.199, 0.201)],
        "cifar100": [(0.507, 0.487, 0.441), (0.267, 0.256, 0.276)],
        "imagenet": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    }[trans_type]
    size = int((256 / 224) * image_size)

    transform = {
        # NCD transform followed UNO
        "ncd": {
            "imagenet": {
                "train": MyTransform(image_size),
                "test": T.Compose([
                        T.Resize(size, interpolation=str_to_pil_interp('bicubic')),
                        # T.Resize((args.resize_size, args.resize_size),interpolation=str_to_pil_interp('bicubic')),
                        T.CenterCrop((image_size, image_size)),
                        T.ToTensor(),
                        T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD))]),
            }
        },
        # GCD transform followed GCD
        "gcd": {
            "imagenet": {
                "train": MyTransform(image_size),
                "test": T.Compose([
                        T.Resize(size, interpolation=str_to_pil_interp('bicubic')),
                        # T.Resize((args.resize_size, args.resize_size),interpolation=str_to_pil_interp('bicubic')),
                        T.CenterCrop((image_size, image_size)),
                        T.ToTensor(),
                        T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD))]),
            }
        },
    }[task][trans_type][mode]

    return transform


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
