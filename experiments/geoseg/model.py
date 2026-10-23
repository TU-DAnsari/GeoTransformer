import torch
import torch.nn as nn
import torch.nn.functional as F

from geotransformer.modules.kpconv import UnaryBlock, LastUnaryBlock
from geotransformer.modules.ops import apply_transform
from geotransformer.modules.geotransformer import GeometricTransformer

from backbone import KPConvBackbone
from loss import compute_segmentation_labels


class GeoSeg(nn.Module):
    def __init__(self, cfg):
        super(GeoSeg, self).__init__()
        self.segmentation_radius = cfg.model.segmentation_radius

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

        # per-point overlap logit from local (backbone) + contextual (transformer) features
        self.seg_head = nn.Sequential(
            UnaryBlock(
                cfg.backbone.output_dim + cfg.geotransformer.output_dim, cfg.seg_head.hidden_dim, cfg.backbone.group_norm
            ),
            LastUnaryBlock(cfg.seg_head.hidden_dim, 1),
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
        with torch.no_grad():
            aligned_src_points = apply_transform(src_points, transform)
            output_dict['ref_labels'] = compute_segmentation_labels(
                ref_points, aligned_src_points, self.segmentation_radius
            )
            output_dict['src_labels'] = compute_segmentation_labels(
                aligned_src_points, ref_points, self.segmentation_radius
            )

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
        ref_feats_t = ref_feats_t.squeeze(0)
        src_feats_t = src_feats_t.squeeze(0)

        output_dict['ref_feats'] = F.normalize(ref_feats_t, p=2, dim=1)
        output_dict['src_feats'] = F.normalize(src_feats_t, p=2, dim=1)

        # 4. Per-point segmentation
        ref_logits = self.seg_head(torch.cat([ref_feats, ref_feats_t], dim=1)).squeeze(1)
        src_logits = self.seg_head(torch.cat([src_feats, src_feats_t], dim=1)).squeeze(1)

        output_dict['ref_logits'] = ref_logits
        output_dict['src_logits'] = src_logits
        output_dict['ref_probs'] = torch.sigmoid(ref_logits)
        output_dict['src_probs'] = torch.sigmoid(src_logits)

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
