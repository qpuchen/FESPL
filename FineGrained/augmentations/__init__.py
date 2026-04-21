from .resnet_transform import get_transforms as get_resnet_transform
from .vit_transform import get_transforms as get_vit_transform


def get_transform(args):
    if "resnet" in args.arch:
        return get_resnet_transform(dataset=args.dataset, args=args)
    elif "vit" in args.arch:
        return get_vit_transform(transform_type="imagenet", args=args)
    else:
        raise NotImplementedError
