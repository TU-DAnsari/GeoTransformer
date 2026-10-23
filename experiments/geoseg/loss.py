import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.loss import WeightedCircleLoss
from geotransformer.modules.ops.transformation import apply_transform
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


class SegmentationLoss(nn.Module):
    def __init__(self, cfg):
        super(SegmentationLoss, self).__init__()
        self.balanced = cfg.seg_loss.balanced

    def _bce(self, logits, labels):
        labels = labels.float()
        pos_weight = None
        if self.balanced:
            num_pos = labels.sum()
            num_neg = labels.numel() - num_pos
            pos_weight = (num_neg / num_pos.clamp(min=1.0)).unsqueeze(0)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)

    def forward(self, output_dict):
        ref_loss = self._bce(output_dict['ref_logits'], output_dict['ref_labels'])
        src_loss = self._bce(output_dict['src_logits'], output_dict['src_labels'])
        return 0.5 * (ref_loss + src_loss)


class CircleLoss(nn.Module):
    def __init__(self, cfg):
        super(CircleLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.circle_loss.positive_margin,
            cfg.circle_loss.negative_margin,
            cfg.circle_loss.positive_optimal,
            cfg.circle_loss.negative_optimal,
            cfg.circle_loss.log_scale,
        )
        self.positive_radius = cfg.circle_loss.positive_radius
        self.negative_radius = cfg.circle_loss.negative_radius

    def forward(self, output_dict, data_dict):
        ref_points = output_dict['ref_points']
        src_points = apply_transform(output_dict['src_points'], data_dict['transform'])
        ref_feats = output_dict['ref_feats']
        src_feats = output_dict['src_feats']

        # point pairs closer than positive_radius should match, pairs beyond negative_radius should not;
        # pairs in between are ambiguous and ignored
        point_dists = pairwise_distance(ref_points, src_points)  # squared, (N, M)
        pos_masks = torch.lt(point_dists, self.positive_radius ** 2)
        neg_masks = torch.gt(point_dists, self.negative_radius ** 2)

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists)

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.seg_loss = SegmentationLoss(cfg)
        self.circle_loss = CircleLoss(cfg)
        self.weight_seg_loss = cfg.loss.weight_seg_loss
        self.weight_circle_loss = cfg.loss.weight_circle_loss

    def forward(self, output_dict, data_dict):
        seg_loss = self.seg_loss(output_dict)
        circle_loss = self.circle_loss(output_dict, data_dict)

        loss = self.weight_seg_loss * seg_loss + self.weight_circle_loss * circle_loss

        return {
            'loss': loss,
            's_loss': seg_loss,
            'c_loss': circle_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.segmentation_threshold = cfg.eval.segmentation_threshold

    @torch.no_grad()
    def forward(self, output_dict, data_dict):
        result_dict = {}
        for prefix in ['ref', 'src']:
            probs = output_dict[f'{prefix}_probs']
            labels = output_dict[f'{prefix}_labels']
            precision, recall, f1, iou = compute_segmentation_metrics(probs, labels, self.segmentation_threshold)
            result_dict[f'{prefix}_P'] = precision
            result_dict[f'{prefix}_R'] = recall
            result_dict[f'{prefix}_F1'] = f1
            result_dict[f'{prefix}_IoU'] = iou

        return result_dict
