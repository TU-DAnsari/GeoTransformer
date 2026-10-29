import torch
import torch.nn as nn

from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.ops.pairwise_distance import pairwise_distance


def compute_segmentation_labels(points, other_points, radius, chunk_size=4096):
    r"""A point is positive if its nearest neighbor in `other_points` is within `radius`.

    Both point clouds must be in the same frame. Chunked to bound memory on large scenes.

    Args:
        points (Tensor): (N, 3)
        other_points (Tensor): (M, 3)

    Returns:
        labels (BoolTensor): (N,)
    """
    labels = []
    for i in range(0, points.shape[0], chunk_size):
        sq_dists = pairwise_distance(points[i : i + chunk_size], other_points)  # (n, M)
        labels.append(torch.lt(sq_dists.min(dim=1)[0], radius ** 2))
    return torch.cat(labels, dim=0)


def compute_segmentation_metrics(probs, labels, threshold=0.5):
    r"""Binary segmentation metrics of per-point matching probabilities against ground-truth labels.

    Args:
        probs (Tensor): (N,)
        labels (BoolTensor): (N,)

    Returns:
        precision, recall, f1, iou (Tensor): scalars, 0 when undefined.
    """
    preds = torch.gt(probs, threshold)
    tp = torch.logical_and(preds, labels).sum().float()
    fp = torch.logical_and(preds, ~labels).sum().float()
    fn = torch.logical_and(~preds, labels).sum().float()
    precision = tp / (tp + fp).clamp(min=1.0)
    recall = tp / (tp + fn).clamp(min=1.0)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp(min=1.0)
    iou = tp / (tp + fp + fn).clamp(min=1.0)
    return precision, recall, f1, iou


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats']
        src_feats = output_dict['src_feats']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)

        loss = self.weight_coarse_loss * coarse_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points'].shape[0]
        src_length_c = output_dict['src_points'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)

        return {'PIR': c_precision}
