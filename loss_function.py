import torch
import math


def cross_entropy(pred, target):
    # return torch.sum( -target * torch.log(pred + 1e-5)) / torch.sum(target)
    pos_term = -target * torch.log(pred + 1e-5) * torch.sum(1-target, dim=1).unsqueeze(1)
    neg_term = -(1-target) * torch.log(1-pred + 1e-5) * torch.sum(target, dim=1).unsqueeze(1)
    return torch.mean(pos_term + neg_term)


class SiameseLoss:
    def __init__(self, margin=0.4):
        self.similiarity_fn = torch.nn.CosineSimilarity(-1, eps=1e-6)
        self.margin = margin

    def cal_similarity(self, anchor, pos, neg):
        if anchor.shape[0] != neg.shape[0]:
            neg = neg.reshape(anchor.shape[0], -1, anchor.shape[-1])
            num_neg_sample = neg.shape[1]
            anchor_tiled = anchor.repeat(num_neg_sample, 1, 1).transpose(0,1)
            neg_similarity = self.similiarity_fn(anchor_tiled, neg)
        else:
            neg_similarity = self.similiarity_fn(anchor, neg)
        if anchor.shape[0] != pos.shape[0]:
            pos = pos.reshape(anchor.shape[0], -1, anchor.shape[-1])
            num_pos_sample = neg.shape[1]
            anchor_tiled = anchor.repeat(num_pos_sample, 1, 1).transpose(0,1)
            pos_similarity = self.similiarity_fn(anchor_tiled, pos)
        else:
            pos_similarity = self.similiarity_fn(anchor, pos)
        return pos_similarity, neg_similarity

    def max_hinge_loss(self, anchor, pos, neg):
        pos_similarity, neg_similarity = self.cal_similarity(anchor, pos, neg)
        # if pos_similarity.shape == neg_similarity:
        #     return torch.mean(torch.max(torch.zeros_like(pos_similarity), self.margin - pos_similarity + neg_similarity))
        # else:
        pos_similarity = torch.mean(pos_similarity, axis=-1)
        neg_similarity = torch.mean(neg_similarity, axis=-1)
        return torch.mean(torch.max(torch.zeros_like(pos_similarity), self.margin - pos_similarity + neg_similarity))
    def __call__(self, anchor, pos, neg):
        return self.max_hinge_loss(anchor, pos, neg)
