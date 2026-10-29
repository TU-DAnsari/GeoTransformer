import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
)
from backbone import KPConvBackbone


class GeoSeg(nn.Module):
    def __init__(self, cfg):
        super(GeoSeg, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = KPConvBackbone(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )


    def forward(self, data_dict):
        output_dict = {}

        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length = data_dict['lengths'][0][0].item()
        points = data_dict['points'][0].detach()
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Ground-truth overlap labels
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points, ref_points, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points, src_points, self.num_points_in_patch
        )

        ref_padded_points = torch.cat([ref_points, torch.zeros_like(ref_points[:1])], dim=0)
        src_padded_points = torch.cat([src_points, torch.zeros_like(src_points[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points,
            src_points,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPConv feature extraction
        feats = self.backbone(feats, data_dict)
        ref_feats = feats[:ref_length]
        src_feats = feats[ref_length:]

        # 3. Geometric self/cross attention over all points
        ref_feats_t, src_feats_t = self.transformer(
            ref_points.unsqueeze(0),
            src_points.unsqueeze(0),
            ref_feats.unsqueeze(0),
            src_feats.unsqueeze(0),
        )
        ref_feats_t_norm = F.normalize(ref_feats_t.squeeze(0), p=2, dim=1)
        src_feats_t_norm = F.normalize(src_feats_t.squeeze(0), p=2, dim=1)

        output_dict['ref_feats'] = ref_feats_t_norm
        output_dict['src_feats'] = src_feats_t_norm

        # 4. Superpoint (= point) correspondences from feature matching, used for evaluation
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_t_norm, src_feats_t_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        return output_dict


def create_model(config):
    model = GeoSeg(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
