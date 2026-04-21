import torch
import torch.nn.functional as F


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.iter = 0

    @torch.no_grad()
    def forward(self, logits):
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes
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


def KD(args, origin_logits, new_logits, mask_lab, T=0.15):
    nlc = args.num_labeled_classes
    origin_logits, new_logits = (
        origin_logits / args.temperature,
        new_logits / args.temperature,
    )
    origin_logits = origin_logits.detach()

    # the prob of unlabeled data on labeled head
    preds = F.softmax(origin_logits[:, :, ~mask_lab] / T, dim=-1)

    pseudo_logits = new_logits[:, :, ~mask_lab]
    pseudo_preds = F.softmax(pseudo_logits[:, :, :, :nlc] / T, dim=-1)

    # gate function to control the weight of kd loss.
    pseudo_preds_all = F.softmax(pseudo_logits / T, dim=-1)
    weight = torch.sum(pseudo_preds_all[:, :, :, :nlc], dim=-1, keepdim=True)

    weight = weight / torch.mean(weight)

    loss_unseen_kd = torch.mean(
        torch.sum(-torch.log(pseudo_preds) * preds * (T**2) * weight, dim=-1),
        dim=[0, 2],
    )

    return loss_unseen_kd
