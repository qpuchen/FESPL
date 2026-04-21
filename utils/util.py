import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as linear_assignment
import random
import os
from config import *
from utils.data_util import init_class_args

import os
import sys
import logging
import functools
from termcolor import colored

def cluster_acc(y_true, y_pred, mask=None):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.stack([ind[0], ind[1]], axis=1)

    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
    old_acc = 0
    new_acc = 0
    if mask is not None:
        ind_map = {j: i for i, j in ind}
        old_classes_gt = set(y_true[mask])
        new_classes_gt = set(y_true[~mask])

        total_old_instances = 0
        if len(old_classes_gt) > 0:
            for i in old_classes_gt:
                old_acc += w[ind_map[i], i]
                total_old_instances += sum(w[:, i])
            old_acc /= total_old_instances

        new_acc = 0
        total_new_instances = 0
        if len(new_classes_gt) > 0:
            for i in new_classes_gt:
                new_acc += w[ind_map[i], i]
                total_new_instances += sum(w[:, i])
            new_acc /= total_new_instances

    return total_acc, old_acc, new_acc


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def info_nce_logits(features, args):
    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    logits = logits / args.temperature
    return logits, labels


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]," "at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits):
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()


class SelfEntropy(torch.nn.Module):
    def __init__(self):
        super(SelfEntropy, self).__init__()

    def forward(self, logits):
        logits_avg = (logits / 0.1).softmax(dim=1).mean(dim=0)
        loss = -F.log_softmax(logits.unsqueeze(dim=1), dim=1)


def scale_mask_softmax(tensor, mask, softmax_dim, scale=1.0):
    # scale = 1.0 if self.opt.dataset != "online_products" else 20.0
    scale_mask_exp_tensor = torch.exp(tensor * scale) * mask.detach()
    scale_mask_softmax_tensor = scale_mask_exp_tensor / (
        1e-8 + torch.sum(scale_mask_exp_tensor, dim=softmax_dim)
    ).unsqueeze(softmax_dim)
    return scale_mask_softmax_tensor


def to_one_hot(self, label, n_dims):
    label_ = label.type(torch.LongTensor).view(-1, 1)
    # label_one_hot = torch.zeros(label.size()[0], n_dims).scatter_(1, label, 1).view(*label.shaoe, -1)
    return torch.zeros(label_.size()[0], n_dims).scatter_(1, label_, 1).view(*label.shape, -1)


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, margin=0.1, alpha=32):
        super().__init__()
        self.num_classes
        self.margin = margin
        self.alpha = alpha

    def forward(self, similarity, label):
        Positive_one_hot = to_one_hot(label, n_dims=self.num_classes)
        Negative_one_hot = 1 - Positive_one_hot

        pos_exp = torch.exp(-self.alpha * (similarity - self.margin))
        neg_exp = torch.exp(self.alpha * (similarity + self.margin))


def init_seed_torch(seed=1):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_configs(args):
    args.temperature = temperature

    # args.image_size = int(image_size)
    args.crop_pct = float(crop_pct)
    args.interpolation = int(interpolation)
    args.trans_type = trans_type
    args.n_views = n_views

    args.num_proxy_base = int(num_proxy_base) if num_proxy_base else None
    args.num_proxy_hard = int(num_proxy_hard) if num_proxy_hard else None
    args.mlp_out_dim = int(mlp_out_dim)
    args.sk_num_iter = int(sk_num_iter)
    args.sk_epsilon = float(sk_epsilon)
    args = init_class_args(args)

    if args.task == "ncd":
        pass
    elif args.task == "gcd":
        args.prop_train_labels = float(prop_train_labels)
    return args


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{dist_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def con_loss(features, labels, margin=0.4):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - margin
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss


def instance_con_loss(features, labels, margin=1.0):
    B, _ = features.shape
    half = B//2
    index = np.arange(half)
    permute_idx = np.random.permutation(B)[:half]
    while (index == permute_idx).sum() != 0 or (index == permute_idx//2).sum() != 0:
        permute_idx = np.random.permutation(B)[:half]
    mask_feats, shuffle_feats = features[:half], features[half:]
    neg_feats = features[permute_idx]
    cos_dis = F.normalize(mask_feats - shuffle_feats)
    neg_dis = F.normalize(mask_feats - neg_feats)
    # cos_matrix_pos = F.normalize(mask_feats - shuffle_feats)
    # cos_matrix_neg = F.normalize(mask_feats - neg_feats)
    loss = cos_dis.sum(dim=-1) - neg_dis.sum(dim=-1) + margin
    # pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    # neg_label_matrix = 1 - pos_label_matrix
    # pos_cos_matrix = 1 - cos_matrix
    # neg_cos_matrix = cos_matrix - margin
    # neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss[loss < 0] = 0
    loss = loss.sum()/B
    return loss