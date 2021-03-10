import torch
import math


def cross_entropy(pred, target):
    # return torch.sum( -target * torch.log(pred + 1e-5)) / torch.sum(target)
    pos_term = -target * torch.log(pred + 1e-5)
    neg_term = -(1-target) * torch.log(1-pred + 1e-5)
    return torch.mean(torch.sum(pos_term + neg_term, dim=-1))


class SiameseLoss:
    def __init__(self, margin=0.4, use_euclid=False):
        if use_euclid:
            # self.similiarity_fn = torch.dist
            def euclidean_sim(x,y):
                distance = torch.dist(x,y)
                num_samples = x.shape[0] * max(x.shape[1], y.shape[1])
                return  (num_samples - distance) / num_samples
            self.similiarity_fn = euclidean_sim 
        else:
            self.similiarity_fn = torch.nn.CosineSimilarity(-1, eps=1e-6)
        self.margin = margin

    def cal_similarity(self, anchor, pos, neg):
        # if anchor.shape[1] != neg.shape[1]:
        #     # neg = neg.reshape(anchor.shape[0], -1, anchor.shape[-1])
        #     num_neg_sample = neg.shape[1]
        #     anchor_tiled = anchor.unsqueeze(1).repeat(1, num_neg_sample, 1, 1).squeeze(2)
        #     neg_similarity = self.similiarity_fn(anchor_tiled, neg)
        # else:
        if len(anchor.shape) ==2:
            anchor = anchor.unsqueeze(1)
        neg_similarity = self.similiarity_fn(anchor, neg)
        # if anchor.shape[1] != pos.shape[1]:
        #     # pos = pos.reshape(anchor.shape[0], -1, anchor.shape[-1])
        #     num_pos_sample = neg.shape[1]
        #     anchor_tiled = anchor.unsqueeze(1).repeat(1, num_neg_sample, 1, 1).squeeze(2)
        #     pos_similarity = self.similiarity_fn(anchor_tiled, pos)
        # else:
        pos_similarity = self.similiarity_fn(anchor, pos)
        return pos_similarity, neg_similarity

    def max_hinge_loss(self, anchor, pos, neg, return_item=False):
        pos_similarity, neg_similarity = self.cal_similarity(anchor, pos, neg)
        # if pos_similarity.shape == neg_similarity:
        #     return torch.mean(torch.max(torch.zeros_like(pos_similarity), self.margin - pos_similarity + neg_similarity))
        # else:
        pos_similarity = torch.mean(pos_similarity, axis=-1)
        neg_similarity = torch.mean(neg_similarity, axis=-1)
        loss = torch.max(torch.zeros_like(pos_similarity), self.margin - pos_similarity + neg_similarity)
        if return_item:
            return loss
        else:
            return torch.mean(loss)
    def __call__(self, anchor, pos, neg, return_item=False):
        return self.max_hinge_loss(anchor, pos, neg, return_item)
