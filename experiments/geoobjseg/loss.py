import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_classification_metrics(probs, labels, threshold=0.5):
    r"""Binary classification metrics of per-point object probabilities against ground-truth labels.

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
    r"""Predator's class-balanced BCE: each class is weighted by the frequency of the other class."""

    def forward(self, output_dict, data_dict):
        logits = output_dict['scene_logits']
        labels = data_dict['labels'].float()

        w_negative = labels.mean()
        w_positive = 1.0 - w_negative
        weights = torch.where(labels > 0.5, w_positive, w_negative)

        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        loss = torch.mean(weights * loss)

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.segmentation_loss = SegmentationLoss()
        self.weight_segmentation_loss = cfg.loss.weight_segmentation_loss

    def forward(self, output_dict, data_dict):
        segmentation_loss = self.segmentation_loss(output_dict, data_dict)

        loss = self.weight_segmentation_loss * segmentation_loss

        return {
            'loss': loss,
            's_loss': segmentation_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.segmentation_threshold = cfg.eval.segmentation_threshold

    @torch.no_grad()
    def forward(self, output_dict, data_dict):
        precision, recall, f1, iou = compute_classification_metrics(
            output_dict['scene_probs'], data_dict['labels'].bool(), self.segmentation_threshold
        )
        return {'P': precision, 'R': recall, 'F1': f1, 'IoU': iou}
