import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def vat_loss_ori(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):
    ul_x = ul_x.unsqueeze(1)
    # find r_adv

    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi * _l2_normalize(d)
        # d = Variable(d.cuda(), requires_grad=True)
        d = Variable(d, requires_grad=True)
        input1 = (ul_x + d).squeeze(1)
        y_hat = model(input1)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        model.zero_grad()

    d = _l2_normalize(d)
    # d = Variable(d.cuda())
    d = Variable(d)
    r_adv = eps * d
    # compute lds
    input2 = (ul_x + r_adv.detach()).squeeze(1)
    y_hat = model(input2)
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl


def _l2_normalize(d):
    # 如果 d 在 GPU 上，将其移到 CPU，并且使用 detach() 去掉梯度信息
    if d.is_cuda:
        d = d.cpu().detach().numpy()
    else:
        d = d.detach().numpy()
    d /= (np.sqrt(np.sum(d**2)) + 1e-8)
    return torch.from_numpy(d).to(get_device())


def vat_loss(model, images, ul_y, xi=1e-6, eps=2.5, num_iters=1, device=None):
    if not device:
        device = get_device()

    ul_x_list = []
    d_list = []

    L = len(images)
    for i in range(L):
        ul_x = images[i].unsqueeze(1).to(device)
        ul_x_list.append(ul_x)
        # 初始化 d_list[i] 并确保 requires_grad=True
        d = torch.randn(ul_x.size(), device=device, requires_grad=True)
        d_list.append(d)

    for _ in range(num_iters):
        input1_list = []
        for i in range(L):
            # 确保 d_list[i] 在每次迭代时都是一个新张量，并且 requires_grad=True
            d_list[i] = xi * _l2_normalize(d_list[i])
            d_list[i].requires_grad_()
            input1_list.append((ul_x_list[i] + d_list[i]).squeeze(1))

        outputs = model.forward(input1_list)
        similarity_base = outputs["similarity_old_base"]
        similarity_base = torch.cat(similarity_base, dim=0)

        indices = torch.topk(similarity_base, k=1).indices.to(model.module.Proxies_old_base_label_one_hot.device)
        y_hat = model.module.Proxies_old_base_label_one_hot[indices].squeeze(dim=1)

        y_hat = y_hat.to(device)
        ul_y = ul_y.to(device)

        delta_kl = kl_div_with_logit(ul_y, y_hat)

        if not delta_kl.requires_grad:
            y_hat.requires_grad_()
            delta_kl = kl_div_with_logit(ul_y, y_hat)
            if not delta_kl.requires_grad:
                raise ValueError("delta_kl does not require grad. Ensure that all inputs have requires_grad=True.")

        delta_kl.backward()

        for i in range(L):
            if d_list[i].grad is not None:
                d_list[i] = d_list[i].grad.data.clone()
            else:
                raise ValueError(f"Gradient for d_list[{i}] is None. Ensure requires_grad is set correctly.")

        model.zero_grad()

    input2_list = []
    for i in range(L):
        d_list[i] = _l2_normalize(d_list[i])
        r_adv = eps * d_list[i].to(device)
        input2_list.append((ul_x_list[i] + r_adv.detach()).squeeze(1))

    outputs = model.forward(input2_list)
    similarity_base = outputs["similarity_old_base"]
    similarity_base = torch.cat(similarity_base, dim=0)

    indices = torch.topk(similarity_base, k=1).indices.to(model.module.Proxies_old_base_label_one_hot.device)
    y_hat = model.module.Proxies_old_base_label_one_hot[indices].squeeze(dim=1)

    delta_kl = kl_div_with_logit(ul_y, y_hat)
    return delta_kl


def kl_div_with_logit(q, p):
    p = p.to(q.device)
    logp = torch.log(p + 1e-8)
    qlogp = (q * logp).sum(dim=1).mean(dim=0)
    qlogq = (q * torch.log(q + 1e-8)).sum(dim=1).mean(dim=0)
    return qlogq - qlogp


def vat_loss_1(model, images, ul_y, xi=1e-6, eps=2.5, num_iters=1, device=None):

    if not device:
        device = get_device()

    ul_x_list = []
    d_list = []

    L = len(images)
    for i in range(L):
        ul_x = images[i].unsqueeze(1)
        ul_x_list.append(ul_x)
        d_list.append(torch.Tensor(ul_x.size()).normal_())

    # ul_x = ul_x.unsqueeze(1)
    # find r_adv

    # d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):

        input1_list = []
        for i in range(L):
            d_list[i] = xi * _l2_normalize(d_list[i]).to(device)
            # d = Variable(d.cuda(), requires_grad=True)
            d_list[i] = Variable(d_list[i], requires_grad=True)
            input1_list.append((ul_x_list[i] + d_list[i]).squeeze(1))

        # d = xi * _l2_normalize(d).to(device)
        # d = Variable(d.cuda(), requires_grad=True)
        # d = Variable(d, requires_grad=True)

        # input1 = (ul_x + d).squeeze(1)
        # y_hat = model(input1)

        outputs = model.forward(input1_list)
        similarity_base = outputs["similarity_old_base"]
        similarity_base = torch.cat(similarity_base, dim=0)
        # y_hat = model.module.Proxies_old_base_label_one_hot[torch.topk(similarity_base, k=1).indices].squeeze(dim=1)
        # # 确保 indices 和 Proxies_old_base_label_one_hot 在同一个设备上
        indices = torch.topk(similarity_base, k=1).indices.to(model.module.Proxies_old_base_label_one_hot.device)
        y_hat = model.module.Proxies_old_base_label_one_hot[indices].squeeze(dim=1)

        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        for i in range(L):
            d_list[i] = d_list[i].grad.data.clone().cpu()

        # d = d.grad.data.clone().cpu()
        model.zero_grad()

    input2_list = []
    for i in range(L):
        d_list[i] = _l2_normalize(d_list[i])
        # d = Variable(d.cuda())
        d_list[i] = Variable(d_list[i])
        r_adv = eps * d_list[i]
        # compute lds
        input2_list.append((ul_x_list[i] + r_adv.detach()).squeeze(1))

    # y_hat = model(input2)
    outputs = model.forward(input2_list)
    similarity_base = outputs["similarity_old_base"]
    # similarity_base = torch.cat(similarity_base, dim=0)
    # pred = model.module.Proxies_old_base_label_one_hot[torch.topk(logits, k=1).indices].squeeze(dim=1)
    # 确保 indices 和 Proxies_old_base_label_one_hot 在同一个设备上
    indices = torch.topk(similarity_base, k=1).indices.to(model.module.Proxies_old_base_label_one_hot.device)
    y_hat = model.module.Proxies_old_base_label_one_hot[indices].squeeze(dim=1)

    # d = _l2_normalize(d)
    # # d = Variable(d.cuda())
    # d = Variable(d)
    # r_adv = eps * d
    # # compute lds
    # input2 = (ul_x + r_adv.detach()).squeeze(1)
    # # y_hat = model(input2)
    # outputs = model.forward(input2)
    # similarity_base = outputs["similarity_old_base"]
    # # similarity_base = torch.cat(similarity_base, dim=0)
    # # pred = model.module.Proxies_old_base_label_one_hot[torch.topk(logits, k=1).indices].squeeze(dim=1)
    # # 确保 indices 和 Proxies_old_base_label_one_hot 在同一个设备上
    # indices = torch.topk(similarity_base, k=1).indices.to(model.module.Proxies_old_base_label_one_hot.device)
    # y_hat = model.module.Proxies_old_base_label_one_hot[indices].squeeze(dim=1)
    #
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl
