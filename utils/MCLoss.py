import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("cuda:0")


class MCRegionLoss(nn.Module):
    def __init__(self, num_classes=196, cnums=5, cgroups=[196], p=0.6, lambda_=1, feat_dim=2048):
        super().__init__()
        if isinstance(cnums, int):
            cnums = [cnums]
        elif isinstance(cnums, tuple):
            cnums = list(cnums)
        assert isinstance(cnums, list), print("Error: cnums should be int or a list of int, not {}".format(type(cnums)))
        assert sum(cgroups) == num_classes, print("Error: num_classes != cgroups.")
        if cgroups[0] == 196 and feat_dim == 2048:
            cgroups = [108, 88]
            cnums = [10, 11]
        elif cgroups[0] == 196 and feat_dim == 768:
            cgroups = [180, 16]
            cnums = [4, 3]
        elif cgroups[0] == 49 and feat_dim == 1024:
            cgroups = [36, 13]
            cnums = [27, 4]

            # cgroups = [196]
            # cnums = [5]
        self.cnums = cnums
        self.cgroups = cgroups
        self.p = p
        self.lambda_ = lambda_
        self.celoss = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def to_one_hot(self, label, n_dims):
        label_ = label.type(torch.LongTensor).view(-1, 1)
        return torch.zeros(label_.size()[0], n_dims).scatter_(1, label_, 1).view(*label.shape, -1)

    def forward(self, feat, model, batch_idx, args):
        label_np = []
        for i in range(len(self.cgroups)):
            if i > 0:
                label_np.append(np.repeat(np.arange(self.cgroups[i]), self.cnums[i]) + self.cgroups[i - 1])
            else:
                label_np.append(np.repeat(np.arange(self.cgroups[i]), self.cnums[i]))
        label = torch.from_numpy(np.concatenate(label_np)).to(feat.device)
        n, c, h, w = feat.size()
        sp = [0]
        tmp = np.array(self.cgroups) * np.array(self.cnums)
        for i in range(len(self.cgroups)):
            sp.append(sum(tmp[: i + 1]))

        mask = self._gen_mask(self.cnums, self.cgroups, self.p).expand_as(feat)
        if feat.is_cuda:
            mask = mask.cuda()
        feature = feat

        feat_group = []
        for i in range(1, len(sp)):
            feat_group.append(feature[:, sp[i - 1] : sp[i]])

        dis_branch = []
        for i in range(len(self.cnums)):
            features = feat_group[i]
            features = features.view(n, -1, h * w)
            dis_branch.append(features)

        # CRA
        dis_branch = torch.cat(dis_branch, dim=1)
        l_dis = torch.stack([self.celoss(dis_branch[d, :, :], label) for d in range(dis_branch.shape[0])]).mean()

        return l_dis

    def _gen_mask(self, cnums, cgroups, p):
        """
        :param cnums:
        :param cgroups:
        :param p: float, probability of random deactivation
        """
        bar = []
        for i in range(len(cnums)):
            foo = np.ones((cgroups[i], cnums[i]), dtype=np.float32).reshape(
                -1,
            )
            drop_num = int(cnums[i] * p)
            drop_idx = []
            for j in range(cgroups[i]):
                drop_idx.append(np.random.choice(np.arange(cnums[i]), size=drop_num, replace=False) + j * cnums[i])
            drop_idx = np.stack(drop_idx, axis=0).reshape(
                -1,
            )
            foo[drop_idx] = 0.0
            bar.append(foo)
        bar = np.hstack(bar).reshape(1, -1, 1, 1)
        bar = torch.from_numpy(bar)

        return bar


class MCRegionLoss_v1(nn.Module):
    def __init__(self, num_classes=256, cnums=8, cgroups=[256], p=0.6, lambda_=1):
        super().__init__()
        if isinstance(cnums, int):
            cnums = [cnums]
        elif isinstance(cnums, tuple):
            cnums = list(cnums)
        assert isinstance(cnums, list), print("Error: cnums should be int or a list of int, not {}".format(type(cnums)))
        assert sum(cgroups) == num_classes, print("Error: num_classes != cgroups.")
        self.cnums = cnums
        self.cgroups = cgroups
        self.p = p
        self.lambda_ = lambda_
        self.celoss = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def to_one_hot(self, label, n_dims):
        label_ = label.type(torch.LongTensor).view(-1, 1)
        return torch.zeros(label_.size()[0], n_dims).scatter_(1, label_, 1).view(*label.shape, -1)

    def forward(self, feat, model, batch_idx, args):
        head = model.module.head_region.to(feat.device)
        label_np = []
        for i in range(len(self.cgroups)):
            if i > 0:
                label_np.append(np.repeat(np.arange(self.cgroups[i]), self.cnums[i]) + self.cgroups[i - 1])
            else:
                label_np.append(np.repeat(np.arange(self.cgroups[i]), self.cnums[i]))
        label = torch.from_numpy(np.concatenate(label_np)).to(feat.device)
        n, c, h, w = feat.size()
        sp = [0]
        tmp = np.array(self.cgroups) * np.array(self.cnums)
        for i in range(len(self.cgroups)):
            sp.append(sum(tmp[: i + 1]))

        mask = self._gen_mask(self.cnums, self.cgroups, self.p).expand_as(feat)
        if feat.is_cuda:
            mask = mask.cuda()
        feature = feat

        feat_group = []
        for i in range(1, len(sp)):
            feat_group.append(feature[:, sp[i - 1] : sp[i]])

        dis_branch = []
        for i in range(len(self.cnums)):
            features = feat_group[i]
            features = features.view(n, -1, h * w)
            dis_branch.append(features)

        # CRA Loss
        dis_branch_logits = head(torch.cat(dis_branch, dim=0))
        l_dis = torch.stack(
            [self.celoss(dis_branch_logits[d, :, :], label) for d in range(dis_branch_logits.shape[0])]
        ).mean()

        return l_dis

    def _gen_mask(self, cnums, cgroups, p):
        """
        :param cnums:
        :param cgroups:
        :param p: float, probability of random deactivation
        """
        bar = []
        for i in range(len(cnums)):
            foo = np.ones((cgroups[i], cnums[i]), dtype=np.float32).reshape(
                -1,
            )
            drop_num = int(cnums[i] * p)
            drop_idx = []
            for j in range(cgroups[i]):
                drop_idx.append(np.random.choice(np.arange(cnums[i]), size=drop_num, replace=False) + j * cnums[i])
            drop_idx = np.stack(drop_idx, axis=0).reshape(
                -1,
            )
            foo[drop_idx] = 0.0
            bar.append(foo)
        bar = np.hstack(bar).reshape(1, -1, 1, 1)
        bar = torch.from_numpy(bar)

        return bar


class MCLoss(nn.Module):
    def __init__(self, num_classes=99, cnums=[20, 21], cgroups=[31, 68], p=0.6, lambda_=5):
        # def __init__(self, num_classes=99, cnums=16, cgroups=128, p=0.6, lambda_=5):
        super().__init__()
        if isinstance(cnums, int):
            cnums = [cnums]
        elif isinstance(cnums, tuple):
            cnums = list(cnums)
        assert isinstance(cnums, list), print("Error: cnums should be int or a list of int, not {}".format(type(cnums)))
        assert sum(cgroups) == num_classes, print("Error: num_classes != cgroups.")

        self.cnums = cnums
        self.cgroups = cgroups
        self.p = p
        self.lambda_ = lambda_
        self.celoss = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.region_label = torch.tensor(np.repeat(np.arange(cgroups), cnums))
        # self.head = nn.Linear(in_features=196, out_features=cgroups)

    def forward(self, feat, targets, model):
        n, c, h, w = feat.size()
        sp = [0]
        tmp = np.array(self.cgroups) * np.array(self.cnums)
        for i in range(len(self.cgroups)):
            sp.append(sum(tmp[: i + 1]))
        # L_div branch
        feature = feat
        feat_group = []
        for i in range(1, len(sp)):
            feat_group.append(
                F.softmax(feature[:, sp[i - 1] : sp[i]].view(n, -1, h * w), dim=2).view(n, -1, h, w)
            )  # Softmax

        l_div = 0.0
        for i in range(len(self.cnums)):
            features = feat_group[i]
            features = F.max_pool2d(
                features.view(n, -1, h * w),
                kernel_size=(self.cnums[i], 1),
                stride=(self.cnums[i], 1),
            )
            l_div = l_div + (1.0 - torch.mean(torch.sum(features, dim=2)) / (self.cnums[i] * 1.0))

        # L_dis branch
        mask = self._gen_mask(self.cnums, self.cgroups, self.p).expand_as(feat)
        if feat.is_cuda:
            mask = mask.cuda()

        feature = mask * feat  # CWA
        # Proxies_local_norm = nn.functional.normalize(model.module.Proxies_local, p=2, dim=0)
        # feature = torch.flatten(feature, 2, -1).matmul(Proxies_local_norm)
        feat_group = []
        for i in range(1, len(sp)):
            feat_group.append(feature[:, sp[i - 1] : sp[i]])

        dis_branch = []
        for i in range(len(self.cnums)):
            features = feat_group[i]
            # features = F.avg_pool2d(features, kernel_size=(self.cnums[i], 1),
            #                         stride=(self.cnums[i], 1))
            features = F.max_pool2d(
                features.view(n, -1, h * w),
                kernel_size=(self.cnums[i], 1),
                stride=(self.cnums[i], 1),
            )
            dis_branch.append(features)
        # dis_branch = torch.matmul(torch.cat(dis_branch, dim=1), model.module.Proxies_local_label_one_hot.to(device))
        # dis_branch = torch.cat([dis_branch[i, targets[i], :].view(1, -1) for i in range(len(targets))], dim=0)
        # torch.matmul(similarity_proxy_old_base, model.module.Proxies_old_base_label_one_hot.to(device))
        # dis_branch: 6 * 620 * 14 * 14 / 6 * 1428 * 14 * 14
        dis_branch = torch.cat(dis_branch, dim=1).view(n, -1, h, w)  # CCMP 6 * 99 * 14 * 14
        dis_branch = self.avgpool(dis_branch).view(n, -1)  # GAP # 6 * 99

        l_dis = self.celoss(dis_branch, targets)

        return l_dis + self.lambda_ * l_div

    def _gen_mask(self, cnums, cgroups, p):
        """
        :param cnums:
        :param cgroups:
        :param p: float, probability of random deactivation
        """
        bar = []
        for i in range(len(cnums)):
            foo = np.ones((cgroups[i], cnums[i]), dtype=np.float32).reshape(
                -1,
            )
            drop_num = int(cnums[i] * p)
            drop_idx = []
            for j in range(cgroups[i]):
                drop_idx.append(np.random.choice(np.arange(cnums[i]), size=drop_num, replace=False) + j * cnums[i])
            drop_idx = np.stack(drop_idx, axis=0).reshape(
                -1,
            )
            foo[drop_idx] = 0.0
            bar.append(foo)
        bar = np.hstack(bar).reshape(1, -1, 1, 1)
        bar = torch.from_numpy(bar)

        return bar


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    mcloss = MCLoss()
    targets = torch.from_numpy(np.arange(2)).long()
    feat = torch.randn((2, 2048, 14, 14))
    loss = mcloss(feat, targets)
    print(loss)
