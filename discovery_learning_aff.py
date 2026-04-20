import argparse
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import SGD
from pl_bolts.optimizers import lr_scheduler

from utils.MCLoss import MCRegionLoss, MCRegionLoss_v1
from utils.util import (
    cluster_acc,
    AverageMeter,
    init_seed_torch,
    init_configs,
    info_nce_logits,
    scale_mask_softmax,
    create_logger,
    con_loss,
    instance_con_loss,
)
from utils.data_util import get_dataloader_gcd, get_dataloader_ncd
from models.model import MultiProxyModel

import wandb
from tqdm import tqdm

import random
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import re


device = torch.device("cuda")
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train_epoch(
    model,
    train_loader,
    test_loader_whole,
    test_loader_unlabel_train,
    args,
    optimizer,
    exp_lr_scheduler,
    criterion_ce,
    criterion_cra,
    epoch,
    swa=False,
    logger=None
):
    loss_record = AverageMeter()
    loss_pc_record = AverageMeter()
    loss_cra_record = AverageMeter()
    loss_pcl_record = AverageMeter()
    model.train()

    for batch_idx, (images, labels, uq_idxs, mask_lab) in enumerate(tqdm(train_loader)):
        images, labels = [image.to(device) for image in images], labels.to(device)
        mask_lab = mask_lab[:, 0]
        mask_lab = mask_lab.to(device).bool()
        outputs = model.forward(images)
        Proxies_old_base_norm = nn.functional.normalize(
            model.module.Proxies_old_base,
            p=2,
            dim=0,
        )
        Proxies_new_base_norm = nn.functional.normalize(
            model.module.Proxies_new_base,
            p=2,
            dim=0,
        )

        # PC Loss
        if mask_lab.sum() > 0:
            similarity_old_base = outputs["similarity_old_base"]
            similarity_old_base = torch.cat(similarity_old_base, dim=0)
            positive_mask_base = torch.eq(
                torch.cat([labels, labels], dim=0).view(-1, 1)
                - model.module.Proxies_old_base_label.view(1, -1).to(device),
                0.0,
            ).float()
            mask_base = torch.zeros_like(similarity_old_base)
            topk_base = math.ceil(args.k * len(args.old_classes) * args.num_proxy_base)
            indices_base = torch.topk(similarity_old_base + 1000 * positive_mask_base, topk_base, dim=1).indices
            mask_base = mask_base.scatter(1, indices_base, 1)
            prob_base = mask_base * similarity_old_base
            logits_base = torch.matmul(prob_base, model.module.Proxies_old_base_label_one_hot.to(device))

            logits_base_mask = 1 - torch.eq(logits_base, 0.0).float().to(device)
            logits_base_softmax = scale_mask_softmax(logits_base, logits_base_mask, 1).to(device)
            loss_pc = torch.stack(
                [
                    (
                        -model.module.to_one_hot(labels[mask_lab], n_dims=len(args.old_classes)).to(device)
                        * torch.log(o[mask_lab] + 1e-20)
                    ).sum()
                    / labels[mask_lab].shape[0]
                    for o in logits_base_softmax.chunk(2)
                ]
            ).mean()
        else:
            loss_pc = torch.Tensor([0])

        # CRA Loss
        feats_map = torch.cat(outputs["feats_map"], dim=0)
        loss_cra = torch.stack([criterion_cra(f, model, batch_idx, args) for f in feats_map.chunk(2)]).mean()

        feats_proj_old = outputs["feats_proj_old"]
        feats_proj_old = torch.cat(feats_proj_old, dim=0)
        feats_proj_old = nn.functional.normalize(feats_proj_old, p=2, dim=1)
        similarity = outputs["similarity_old_base"]
        similarity = torch.cat(similarity, dim=0)

        # PCL Loss
        if args.cl_version == "pcl":
            con_feats = torch.cat(
                [
                    torch.cat([f, Proxies_old_base_norm.T, Proxies_new_base_norm.T], dim=0)
                    for f in feats_proj_old.chunk(2)
                ]
            )
        elif args.cl_version == "vcl":
            con_feats = feats_proj_old
        con_logits, con_labels = info_nce_logits(features=con_feats, args=args)
        loss_pcl = criterion_ce(con_logits, con_labels)

        # Ablation ce
        # if mask_lab.sum() > 0:
        #     logits = outputs['logits_lab']
        #     loss_ce = torch.stack([criterion_ce(l[mask_lab], labels[mask_lab]) for l in logits]).mean()

        loss_basic = args.weight_contra * loss_pcl + args.weight_local * loss_cra
        if mask_lab.sum() > 0 and (~mask_lab).sum() > 0:
            loss = loss_basic + args.weight_global * loss_pc
            # loss = 2 * loss_ce + 1 * loss_pcl
        elif (~mask_lab).sum() == 0:
            loss = loss_basic + args.weight_global * loss_pc
            # loss = 2 * loss_ce + 1 * loss_pcl
        elif mask_lab.sum() == 0:
            loss = loss_basic
            # loss = args.weight_contra * loss_pcl + args.weight_local * loss_cra
            # loss = 1 * loss_pcl

        if args.weight_con > 0:
            feats_ = torch.cat(outputs["feats_"], dim=0)
            labels_ = torch.cat([labels, labels], dim=0)
            MARGIN = 1
            # loss_con = instance_con_loss(feats_, labels_, MARGIN)
            loss_con = con_loss(feats_, labels_)
            loss = loss + args.weight_con * loss_con

        loss_record.update(loss.item(), images[0].size(0))
        loss_pc_record.update(loss_pc.item(), images[0].size(0))
        loss_pcl_record.update(loss_pcl.item(), images[0].size(0))
        loss_cra_record.update(loss_cra.item(), images[0].size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if swa:
            exp_lr_scheduler.step()
    # print(
    #     "Train Epoch: {} Avg Loss: {:.4f} Avg PC Loss: {:.4f} Avg CRA Loss: {:.4f} Avg PCL Loss: {:.4f} Lr: {}".format(
    #         epoch,
    #         loss_record.avg,
    #         loss_pc_record.avg,
    #         loss_cra_record.avg,
    #         loss_pcl_record.avg,
    #         optimizer.param_groups[0]["lr"],
    #     )
    # )
    logger.info(
        "Train Epoch: {} Avg Loss: {:.4f} Avg PC Loss: {:.4f} Avg CRA Loss: {:.4f} Avg PCL Loss: {:.4f} Lr: {}".format(
            epoch,
            loss_record.avg,
            loss_pc_record.avg,
            loss_cra_record.avg,
            loss_pcl_record.avg,
            optimizer.param_groups[0]["lr"],
        )
    )
    # wandb.log(
    #     {
    #         "train/epoch": epoch,
    #         "train/loss avg": loss_record.avg,
    #         "train/loss PC": loss_pc_record.avg,
    #         "train/loss CRA": loss_cra_record.avg,
    #         "train/loss PCL": loss_pcl_record.avg,
    #         "train/lr": optimizer.param_groups[0]["lr"],
    #     }
    # )
    with torch.no_grad():
        # print("Test on test whole classes")
        logger.info("Test on test whole classes")
        test(model, test_loader_whole, args, test_split="test/all", logger=logger)

        # print("Test on train unlabel classes")
        logger.info("Test on train unlabel classes")
        test(model, test_loader_unlabel_train, args, test_split="train/new", logger=logger)
    if not swa:
        exp_lr_scheduler.step()
    else:
        save_path = os.path.join(args.swa_save_dir, "epoch_" + str(epoch) + ".pth")
        torch.save(model.state_dict(), save_path)
        # print("model save to {}.".format(save_path))
        logger.info("model save to {}.".format(save_path))


def train(model, train_loader, test_loader_whole, test_loader_unlabel_train, args, logger=None):
    optimizer = SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    exp_lr_scheduler = lr_scheduler.LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=10,
        max_epochs=args.epochs,
        warmup_start_lr=args.min_lr,
        eta_min=args.min_lr,
    )
    criterion_ce = nn.CrossEntropyLoss()

    if args.local == "v1":
        criterion_mcregionloss = MCRegionLoss_v1(
            num_classes=args.cgroups,
            cnums=int(model.module.feat_dim / args.cgroups),
            cgroups=[args.cgroups],
        )
    elif args.local == "v2":
        criterion_mcregionloss = MCRegionLoss(
            num_classes=args.cgroups,
            cnums=int(model.module.feat_dim / args.cgroups),
            cgroups=[args.cgroups],
            feat_dim=args.hidden_dim
        )

    for epoch in range(args.epochs):
        train_epoch(
            model=model,
            train_loader=train_loader,
            test_loader_whole=test_loader_whole,
            test_loader_unlabel_train=test_loader_unlabel_train,
            args=args,
            optimizer=optimizer,
            exp_lr_scheduler=exp_lr_scheduler,
            criterion_ce=criterion_ce,
            criterion_cra=criterion_mcregionloss,
            epoch=epoch,
            swa=False,
            logger=logger
        )
    torch.save(model.state_dict(), args.model_save_path)
    # print("model save to {}.".format(args.model_save_path))
    logger.info("model save to {}.".format(args.model_save_path))


def test(model, test_loader, args, center=None, test_split="", logger=None):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    all_feats = []
    targets = np.array([])
    mask = np.array([])

    # print("Collating features...")
    logger.info("Collating features...")

    # First extract all features
    model.eval()
    for images, labels, _ in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)

        # Pass features through base model and then additional learnable transform (linear layer)
        outputs = model.forward(images)
        feats = outputs["feats_proj_old"]
        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, labels.cpu().numpy())
        mask = np.append(
            mask,
            np.array([True if x.item() in range(len(args.old_classes)) else False for x in labels]),
        ).astype(bool)
    # -----------------------
    # K-MEANS
    # -----------------------
    # print('Fitting K-Means...')
    logger.info('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=int(targets.max() + 1), random_state=0, n_init="auto").fit(all_feats)
    preds = kmeans.labels_
    acc = cluster_acc(targets.astype(int), preds.astype(int), mask.astype(bool))
    # print(
    #     "Test all kmeans acc {:.4f}, old acc {:.4f}, new acc {:.4f}".format(
    #         acc[0],
    #         acc[1],
    #         acc[2],
    #     )
    # )
    logger.info(
        "Test all kmeans acc {:.4f}, old acc {:.4f}, new acc {:.4f}".format(
            acc[0],
            acc[1],
            acc[2],
        )
    )
    # wandb.log(
    #     {
    #         f"{test_split}/kmeans all acc": acc[0],
    #         f"{test_split}/kmeans old acc": acc[1],
    #         f"{test_split}/kmeans new acc": acc[2],
    #     }
    # )

    return kmeans.cluster_centers_


import re


def extract_values(filename):
    # 定义正则表达式模式
    pattern = r"pc([0-9.]+)-cra([0-9.]+)-reg([0-9.]+)-con([0-9.]+)-adv(\w+)-un(\w+)"
    match = re.search(pattern, filename)

    if match:
        pc_value = float(match.group(1))
        cra_value = float(match.group(2))
        reg_value = float(match.group(3))
        con_value = float(match.group(4))
        adv_value = match.group(5)
        un_value = match.group(6)
        return pc_value, cra_value, reg_value, con_value, adv_value, un_value
    else:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Supervised learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--min_lr", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument(
        "--dataset",
        type=str,
        default="SoyAgeing-R1",
        help="options: SoyAgeing-{R1,R3,R4,R5,R6}, SoyGene, SoyGlobal, SoyLocal, Cotton",
    )
    parser.add_argument("--multicrop", default=False, action="store_true", help="activates multicrop")
    parser.add_argument("--num_crops", type=int, default=2, help="number of multicrop")
    parser.add_argument("--task", type=str, default="ncd", help="options:ncd, gcd")
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="supervised-SoyAgeing-R1-wg2.0-wl1.0-wp1.0-k0.05.pth",
    )
    parser.add_argument(
        "--train",
        default=True,
        type=str2bool,
        help="train or test",
    )
    parser.add_argument("--cgroups", type=int, default=196)
    parser.add_argument("--weight_con", type=float, default=1.0)
    parser.add_argument("--weight_global", type=float, default=1.0)
    parser.add_argument("--weight_local", type=float, default=0.6)
    parser.add_argument("--weight_contra", type=float, default=0.8)
    parser.add_argument("--weight_sup_con", type=float, default=0)
    parser.add_argument("--weight_relax", type=float, default=0)
    parser.add_argument("--local", type=str, default="v2")
    parser.add_argument("--proxy_mode", type=str, default="fix_old")
    parser.add_argument("--model_dataset", type=str, default="SoyAgeing-R1")
    parser.add_argument("--cl_version", type=str, default="pcl", choices=["pcl", "vcl"])

    parser.add_argument("--k", type=float, default=0.05)

    parser.add_argument("--image_size", type=int, default=448,
                        help="the size of the image")
    parser.add_argument("--hidden_dim", type=int, default=2048,
                        help="dimensions of hidden layers")

    parser.add_argument('--local-rank', type=int, default=0)
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    parser.add_argument('--adversarial_training', action='store_true', help='Whether to use adversarial training?')


    from affnet import modeling_arguments
    from affnet.utils import parser_to_opts

    # cvnet arguments, including models
    parser_model = modeling_arguments(parser=parser)
    ops = parser_to_opts(parser_model)

    args = parser.parse_args()
    args = init_configs(args)  # init configs of data augmentation and model
    # init_seed_torch(args.seed)


    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("args.pretrained: ", args.pretrained)
    weight_pc, weight_cra, weight_reg, weight_con, adv_value, un_value = extract_values(args.pretrained)
    task_name = "-".join(
        [
            args.task,
            args.model,
            args.dataset,
            f"pc{weight_pc}",
            f"cra{weight_cra}",
            f"reg{weight_reg}",
            f"con{weight_con}",
            f"adv{adv_value}",
            f"un{un_value}",
        ]
    )

    # pretrained_dir = os.path.join(args.output_dir, args.model, "supervised_learning")
    # args.pretrained_path = os.path.join(pretrained_dir, args.pretrained)
    pretrained_dir = os.path.join(args.output_dir, task_name, "supervised_learning_aff")
    args.pretrained_path = os.path.join(pretrained_dir, args.pretrained)


    runner_name = os.path.basename(__file__).split(".")[0]
    # model_save_dir = os.path.join(args.output_dir, args.model, runner_name)

    run_name = "-".join(
        [
            args.task,
            args.model,
            "discover",
            args.dataset,
            args.proxy_mode,
            f"pc{args.weight_global}",
            f"cra{args.weight_local}",
            f"pcl{args.weight_contra}",
            f"con{args.weight_con}",
            f"adv{args.adversarial_training}",
        ]
    )

    task_dl_name = "-".join(
        [
            args.task,
            args.model,
            args.dataset,
            args.proxy_mode,
            f"pc{args.weight_global}",
            f"cra{args.weight_local}",
            f"pcl{args.weight_contra}",
            f"con{args.weight_con}",
            f"adv{args.adversarial_training}",
        ]
    )

    model_save_dir = os.path.join(pretrained_dir, task_dl_name, runner_name)
    os.makedirs(model_save_dir, exist_ok=True)
    args.model_save_path = model_save_dir + "/" + "{}.pth".format(run_name)

    logger = create_logger(model_save_dir, name=f"{run_name}")

    # wandb.init(
    #     project="ncd",
    #     name=run_name,
    #     config=vars(args),
    # )
    model = MultiProxyModel(
        model=args.model,
        num_old_classes=len(args.old_classes),
        num_new_classes=len(args.new_classes),
        num_proxy_base=args.num_proxy_base,
        num_proxy_hard=args.num_proxy_hard,
        mlp_out_dim=args.mlp_out_dim,
        hidden_dim=args.hidden_dim,
        mode="supervised",
        cgroups=args.cgroups,
    )
    model = torch.nn.DataParallel(model)
    if args.pretrained_path:
        # print("load from {}".format(args.pretrained_path))
        logger.info("load from {}".format(args.pretrained_path))
        state_dict = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model.to(device)

    if args.task == "gcd":
        train_loader, test_loader_whole, test_loader_unlabel_train = get_dataloader_gcd(
            args.dataset,
            args,
            mode="discover",
        )
        with torch.no_grad():
            Proxy_new = test(model, test_loader_unlabel_train, args, logger=logger)
            # Proxy_old = test(model, train_loader_test, args, logger=logger)
        model.module.Proxies_new_base.data = copy.deepcopy(
            F.normalize(torch.from_numpy(Proxy_new), p=2, dim=0).T.to(device),
        )
    elif args.task == "ncd":
        (
            train_loader,
            test_loader_unlabel_train,
            test_loader_label,
            test_loader_unlabel,
            test_loader_whole,
        ) = get_dataloader_ncd(args.dataset, args, mode="unsupervised")

        train_loader_label, _ = get_dataloader_ncd(args.dataset, args, mode="supervised")
        train_loader_test = copy.deepcopy(train_loader_label)
        train_loader_test.dataset.transform = test_loader_whole.dataset.transform
        with torch.no_grad():
            Proxy_new = test(model, test_loader_unlabel_train, args, logger=logger)
        model.module.Proxies_new_base.data = copy.deepcopy(
            F.normalize(torch.from_numpy(Proxy_new), p=2, dim=0).T.to(device),
        )

    for name, param in model.named_parameters():
        if args.proxy_mode == "fix_old":
            if "Proxies_old" in name:
                param.requires_grad = False
        elif args.proxy_mode == "fix_new":
            if "Proxies_new" in name:
                param.requires_grad = False
        elif args.proxy_mode == "fix_all":
            if "Proxies_old" in name or "Proxies_new" in name:
                param.requires_grad = False
        else:
            break

    if args.train:
        train(
            model,
            train_loader=train_loader,
            test_loader_whole=test_loader_whole,
            test_loader_unlabel_train=test_loader_unlabel_train,
            args=args,
            logger=logger
        )
    model.load_state_dict(torch.load(args.model_save_path))

    with torch.no_grad():
        # print("test on unlabeled train classes")
        logger.info("test on unlabeled train classes")
        test(
            model,
            test_loader_unlabel_train,
            args,
            test_split="origin/train/new",
            logger = logger
        )

        # print("test on all classes")
        logger.info("test on all classes")
        test(
            model,
            test_loader_whole,
            args,
            test_split="origin/test/all",
            logger=logger
        )

        # print("test on labeled classes")
        logger.info("test on labeled classes")
        test(model, test_loader_label, args, test_split="origin/test/old", logger=logger)

        # print("test on unlabeled test classes")
        logger.info("test on unlabeled test classes")
        test(model, test_loader_unlabel, args, test_split="origin/test/new", logger=logger)

    # print("discover training end")
    logger.info("discover training end")
