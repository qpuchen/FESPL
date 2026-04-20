import argparse
import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
from pl_bolts.optimizers import lr_scheduler
from sklearn.cluster import KMeans
from torch.optim import SGD
from tqdm import tqdm

import wandb
from models.model import MultiProxyModel
from utils.data_util import get_dataloader_gcd, get_dataloader_ncd
from utils.MCLoss import MCRegionLoss
from utils.util import (
    AverageMeter,
    cluster_acc,
    init_configs,
    init_seed_torch,
    scale_mask_softmax,
    create_logger,
    con_loss,
    instance_con_loss,
)

import random
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from utils.losses import vat_loss, edl_digamma_loss, edl_log_loss, edl_mse_loss

device = torch.device("cuda")
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

def train(model, train_loader, test_loader, args, logger=None):
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
    criterion_cra = MCRegionLoss(
        num_classes=args.cgroups,
        cnums=int(model.module.feat_dim / args.cgroups),
        cgroups=[args.cgroups],
        feat_dim=args.hidden_dim
    )

    train_loader_test = copy.deepcopy(train_loader)
    train_loader_test.dataset.transform = test_loader.dataset.transform
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        loss_pc_record = AverageMeter()
        loss_reg_record = AverageMeter()
        loss_cra_record = AverageMeter()
        model.train()
        for batch_idx, (images, labels, _) in enumerate(tqdm(train_loader)):
            images, labels = [image.to(device) for image in images], labels.to(device)
            outputs = model.forward(images)

            # PC Loss
            similarity_base = outputs["similarity_old_base"]
            similarity_base = torch.cat(similarity_base, dim=0)
            positive_mask_base = torch.eq(
                torch.cat([labels, labels], dim=0).view(-1, 1)
                - model.module.Proxies_old_base_label.view(1, -1).to(device),
                0.0,
            ).float()
            mask_base = torch.zeros_like(similarity_base)
            topk_base = math.ceil(args.k * len(args.old_classes) * args.num_proxy_base)
            indices_base = torch.topk(
                similarity_base + 1000 * positive_mask_base,
                topk_base,
                dim=1,
            ).indices
            mask_base = mask_base.scatter(1, indices_base, 1)
            prob_base = mask_base * similarity_base
            logits_base = torch.matmul(
                prob_base,
                model.module.Proxies_old_base_label_one_hot.to(device),
            )
            logits_base_mask = 1 - torch.eq(logits_base, 0.0).float().to(device)
            logits_base_softmax = scale_mask_softmax(
                logits_base,
                logits_base_mask,
                1,
            ).to(device)
            loss_pc = torch.stack(
                [
                    (
                        -model.module.to_one_hot(labels, n_dims=len(args.old_classes)).to(device) * torch.log(o + 1e-20)
                    ).sum()
                    / labels.shape[0]
                    for o in logits_base_softmax.chunk(2)
                ]
            ).mean()

            # REG Loss
            Proxies_old_base_norm = nn.functional.normalize(
                model.module.Proxies_old_base,
                p=2,
                dim=0,
            )
            similarity_proxy_old_base = Proxies_old_base_norm.t().matmul(Proxies_old_base_norm)
            similarity_proxy_logits_old_base = torch.matmul(
                similarity_proxy_old_base,
                model.module.Proxies_old_base_label_one_hot.to(device),
            )
            loss_reg = criterion_ce(
                similarity_proxy_logits_old_base,
                model.module.Proxies_old_base_label.to(device),
            )

            # CRA Loss
            feats_map = torch.cat(outputs["feats_map"], dim=0)
            loss_cra = torch.stack([criterion_cra(f, model, batch_idx, args) for f in feats_map.chunk(2)]).mean()
            loss = args.weight_pc * loss_pc + args.weight_reg * loss_reg + args.weight_cra * loss_cra

            feats_ = torch.cat(outputs["feats_"], dim=0)
            labels_ = torch.cat([labels, labels], dim=0)
            if args.weight_con > 0:

                MARGIN = 1
                # loss_con = instance_con_loss(feats_, labels_, MARGIN)
                loss_con = con_loss(feats_, labels_)
                loss = loss + args.weight_con * loss_con


            # pred = model.module.Proxies_old_base_label_one_hot[torch.topk(logits, k=1).indices].squeeze(dim=1)
            # 确保 indices 和 Proxies_old_base_label_one_hot 在同一个设备上
            indices = torch.topk(similarity_base, k=1).indices.to(model.module.Proxies_old_base_label_one_hot.device)
            pred = model.module.Proxies_old_base_label_one_hot[indices].squeeze(dim=1)
            # _, pred = pred.max(1)

            y = one_hot_embedding(labels, len(args.old_classes))
            y_ = torch.cat([y, y], dim=0).float()
            # pred_ = torch.argmax(pred, dim=1)

            if args.adversarial_training:
                v_loss = vat_loss(model, images, pred, eps=2.5, device=device) # model.module.encoder
                loss = loss + v_loss
            if args.use_uncertainty == 'edl_digamma_loss':
                digamma_loss = edl_digamma_loss(pred, y_, epoch, len(args.old_classes), 2, device)
                loss = loss + digamma_loss
            elif args.use_uncertainty == 'edl_log_loss':
                log_loss = edl_log_loss(pred, y_.float(), epoch, len(args.old_classes), 2, device)
                loss = loss + log_loss
            elif args.use_uncertainty == 'edl_mse_loss':
                mse_loss = edl_mse_loss(pred, y_.float(), epoch, len(args.old_classes), 2, device)
                loss = loss + mse_loss

            loss_record.update(loss.item(), images[0].size(0))
            loss_pc_record.update(loss_pc.item(), images[0].size(0))
            loss_reg_record.update(loss_reg.item(), images[0].size(0))
            loss_cra_record.update(loss_cra.item(), images[0].size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(
        #     "Train Epoch: {} Avg Loss: {:.4f} Loss PC : {:.4f} Loss REG: {:.4f} Loss CRA : {:.4f} Lr: {} Lr: {}".format(
        #         epoch,
        #         loss_record.avg,
        #         loss_pc_record.avg,
        #         loss_reg_record.avg,
        #         loss_cra_record.avg,
        #         optimizer.param_groups[0]["lr"],
        #         optimizer.param_groups[0]["lr"],
        #     )
        # )
        logger.info(
            "Train Epoch: {} Avg Loss: {:.4f} Loss PC : {:.4f} Loss REG: {:.4f} Loss CRA : {:.4f} Lr: {} Lr: {}".format(
                epoch,
                loss_record.avg,
                loss_pc_record.avg,
                loss_reg_record.avg,
                loss_cra_record.avg,
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[0]["lr"],
            )
        )

        # wandb.log(
        #     {
        #         "train/epoch": epoch,
        #         "train/loss avg": loss_record.avg,
        #         "train/loss PC": loss_pc_record.avg,
        #         "train/loss REG": loss_reg_record.avg,
        #         "train/loss CRA": loss_cra_record.avg,
        #         "train/lr": optimizer.param_groups[0]["lr"],
        #     }
        # )
        with torch.no_grad():
            # print("Test on test label classes")
            logger.info("Test on test label classes")
            test(model, test_loader, args, test_split="test/old", logger=logger)
        exp_lr_scheduler.step()


def test(model, test_loader, args, test_split="", logger=None):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    for images, labels, _ in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model.forward(images)
        logits = outputs["similarity_old_base"]
        # pred = model.module.Proxies_old_base_label_one_hot[torch.topk(logits, k=1).indices].squeeze(dim=1)
        # 确保 indices 和 Proxies_old_base_label_one_hot 在同一个设备上
        indices = torch.topk(logits, k=1).indices.to(model.module.Proxies_old_base_label_one_hot.device)
        pred = model.module.Proxies_old_base_label_one_hot[indices].squeeze(dim=1)

        _, pred = pred.max(1)
        targets = np.append(targets, labels.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

        acc = cluster_acc(
            targets.astype(int),
            preds.astype(int),
        )
    # print("Test proxy acc {:.4f}".format(acc[0]))
    logger.info("Test proxy acc {:.4f}".format(acc[0]))
    # wandb.log({f"{test_split}/global acc": acc[0]})

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

    kmeans = KMeans(n_clusters=int(targets.max() + 1), random_state=0, n_init="auto").fit(all_feats)
    preds = kmeans.labels_
    acc = cluster_acc(targets.astype(int), preds.astype(int), mask.astype(bool))
    # print(
    #     "Test kmeans all acc {:.4f}, old acc {:.4f}, new acc {:.4f}".format(
    #         acc[0],
    #         acc[1],
    #         acc[2],
    #     )
    # )
    logger.info(
        "Test kmeans all acc {:.4f}, old acc {:.4f}, new acc {:.4f}".format(
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
    return preds


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
    parser.add_argument(
        "--multicrop",
        default=False,
        action="store_true",
        help="activates multicrop",
    )
    parser.add_argument("--num_crops", type=int, default=2, help="number of multicrop")
    parser.add_argument("--task", type=str, default="ncd", help="options:ncd, gcd")
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument(
        "--train",
        default=True,
        action="store_true",
        help="train or test",
    )
    parser.add_argument("--cgroups", type=int, default=196)
    parser.add_argument("--weight_con", type=float, default=1.0)
    parser.add_argument("--weight_pc", type=float, default=2.0)
    parser.add_argument("--weight_cra", type=float, default=0.6)
    parser.add_argument("--weight_reg", type=float, default=1.0)
    parser.add_argument("--weight_cra_dis", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=0.05)

    parser.add_argument("--image_size", type=int, default=448,
                        help="the size of the image")
    parser.add_argument("--hidden_dim", type=int, default=2048,
                        help="dimensions of hidden layers")

    parser.add_argument('--local-rank', type=int, default=0)
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    parser.add_argument('--adversarial_training', action='store_true', help='Whether to use adversarial training?')
    parser.add_argument('--use_uncertainty', '-uu', type=str, default='edl_mse_loss',
                        help='Which loss is used or only use cross entropy?  [edl_digamma_loss, edl_log_loss, edl_mse_loss]')


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

    runner_name = os.path.basename(__file__).split(".")[0]

    run_name = "-".join(
        [
            args.task,
            args.model,
            "supervised",
            args.dataset,
            f"pc{args.weight_pc}",
            f"cra{args.weight_cra}",
            f"reg{args.weight_reg}",
            f"con{args.weight_con}",
            f"adv{args.adversarial_training}",
            f"un{args.use_uncertainty}",
        ]
    )

    task_name = "-".join(
        [
            args.task,
            args.model,
            args.dataset,
            f"pc{args.weight_pc}",
            f"cra{args.weight_cra}",
            f"reg{args.weight_reg}",
            f"con{args.weight_con}",
            f"adv{args.adversarial_training}",
            f"un{args.use_uncertainty}",
        ]
    )

    model_save_dir = os.path.join(args.output_dir, task_name, runner_name)
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
        ops=ops
    )
    if args.task == "gcd":
        train_loader, test_loader_label = get_dataloader_gcd(
            args.dataset,
            args,
            mode="supervised",
        )
    elif args.task == "ncd":
        train_loader, test_loader_label = get_dataloader_ncd(
            args.dataset,
            args,
            mode="supervised",
        )

    if args.train:
        # model = torch.nn.DataParallel(model)

        model.to(device)

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          broadcast_buffers=False, find_unused_parameters=True)

        train(
            model,
            train_loader=train_loader,
            test_loader=test_loader_label,
            args=args,
            logger=logger,
        )
        torch.save(model.state_dict(), args.model_save_path)
        # print("model save to {}.".format(args.model_save_path))
        logger.info("model save to {}.".format(args.model_save_path))

    state_dict = torch.load(args.model_save_path, map_location=device)
    if not args.train:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    with torch.no_grad():
        # print("test on labeled classes")
        logger.info("test on labeled classes")
        test(model, test_loader_label, args, test_split="test/old", logger=logger)

        train_loader_test = copy.deepcopy(train_loader)
        train_loader_test.dataset.transform = test_loader_label.dataset.transform
        # print("Test on train label classes")
        logger.info("Test on train label classes")
        test(model, train_loader_test, args, test_split="train/old", logger=logger)

    # print("supervised training end")
    logger.info("supervised training end")
