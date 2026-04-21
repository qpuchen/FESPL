from config import dino_pretrain_path
from .resnet import ResNet
from .vision_transformer import *
import torch


def get_backbone(args, **kwargs):
    if "resnet" in args.arch:
        return _get_resnet_backbone(args)
    elif "vit" in args.arch:
        return _get_vit_backbone(args, **kwargs)
    else:
        raise NotImplementedError("The arch has not implemented.")


def _get_resnet_backbone(args):
    backbone = ResNet(args.arch, args.low_res)
    return backbone


def _get_vit_backbone(args, drop_path_rate=0):
    vit_backbone = vit_base(drop_path_rate=drop_path_rate)
    # vit_backbone = vit_small(drop_path_rate=drop_path_rate)
    try:
        state_dict = torch.load(dino_pretrain_path, map_location="cpu")
        weight = {k.replace("module.backbone.", ""): v for k, v in state_dict["student"].items()}
        vit_backbone.load_state_dict(weight, strict=False)
    except RuntimeError:
        print("Noting you are failed to load the pretrained model, we will load the dino pretrained model.")
    for m in vit_backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in vit_backbone.named_parameters():
        if "block" in name:
            block_num = int(name.split(".")[1])
            if block_num >= args.vit_layers:
                m.requires_grad = True

    return vit_backbone
