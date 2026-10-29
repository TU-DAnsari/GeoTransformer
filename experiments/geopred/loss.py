import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_classification_metrics(probs, labels, threshold=0.5):
    r"""Binary classification metrics of per-point overlap probabilities against ground-truth labels.

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


class OverlapLoss(nn.Module):
    r"""Predator's class-balanced BCE: each class is weighted by the frequency of the other class."""

    def forward(self, output_dict):
        logits = torch.cat([output_dict['ref_overlap_logits'], output_dict['src_overlap_logits']], dim=0)
        labels = torch.cat([output_dict['ref_overlap_labels'], output_dict['src_overlap_labels']], dim=0).float()

        w_negative = labels.mean()
        w_positive = 1.0 - w_negative
        weights = torch.where(labels > 0.5, w_positive, w_negative)

        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        loss = torch.mean(weights * loss)

        return loss


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.overlap_loss = OverlapLoss()
        self.weight_overlap_loss = cfg.loss.weight_overlap_loss

    def forward(self, output_dict, data_dict):
        overlap_loss = self.overlap_loss(output_dict)

        loss = self.weight_overlap_loss * overlap_loss

        return {
            'loss': loss,
            'o_loss': overlap_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.overlap_threshold = cfg.eval.overlap_threshold

    @torch.no_grad()
    def forward(self, output_dict, data_dict):
        result_dict = {}
        for prefix in ['ref', 'src']:
            precision, recall, f1, iou = compute_classification_metrics(
                output_dict[f'{prefix}_overlap_probs'],
                output_dict[f'{prefix}_overlap_labels'],
                self.overlap_threshold,
            )
            result_dict[f'{prefix}_P'] = precision
            result_dict[f'{prefix}_R'] = recall
            result_dict[f'{prefix}_F1'] = f1
            result_dict[f'{prefix}_IoU'] = iou

        return result_dict
