import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import apply_transform, pairwise_distance
from geotransformer.modules.geotransformer import GeometricTransformer

from backbone import KPConvEncoder, OverlapDecoder


@torch.no_grad()
def compute_overlap_labels(points, other_points, radius, chunk_size=4096):
    r"""A point is in the overlap region if its nearest neighbor in `other_points` is within `radius`.

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


class GeoPredator(nn.Module):
    r"""KPConv encoder -> GeoTransformer self/cross attention on the coarse nodes -> Predator overlap head.

    No correspondences or transformation are estimated: the only output is the per-point probability that a point of
    one cloud lies in the overlap region with the other cloud.
    """

    def __init__(self, cfg):
        super(GeoPredator, self).__init__()
        self.overlap_radius = cfg.model.overlap_radius
        self.cross_score = cfg.overlap.cross_score

        self.encoder = KPConvEncoder(
            cfg.backbone.input_dim,
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

        # Predator: coarse overlap score from the conditioned features, optionally cross-attended to the other cloud
        self.proj_score = nn.Linear(cfg.geotransformer.output_dim, 1)
        if self.cross_score:
            self.epsilon = nn.Parameter(torch.tensor(-5.0))

        decoder_input_dim = cfg.geotransformer.output_dim + (2 if self.cross_score else 1)
        self.decoder = OverlapDecoder(decoder_input_dim, cfg.backbone.init_dim, cfg.backbone.group_norm)

    def forward(self, data_dict):
        output_dict = {}

        feats = data_dict['features'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Ground-truth overlap labels (only when the ground-truth transformation is available)
        if 'transform' in data_dict:
            transform = data_dict['transform'].detach()
            aligned_src_points = apply_transform(src_points, transform)
            output_dict['ref_overlap_labels'] = compute_overlap_labels(
                ref_points, aligned_src_points, self.overlap_radius
            )
            output_dict['src_overlap_labels'] = compute_overlap_labels(
                aligned_src_points, ref_points, self.overlap_radius
            )

        # 2. KPConv Encoder
        encoder_feats_list = self.encoder(feats, data_dict)
        feats_c = encoder_feats_list[-1]

        # 3. Geometric Transformer (self + cross attention)
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c = ref_feats_c.squeeze(0)
        src_feats_c = src_feats_c.squeeze(0)

        # 4. Coarse overlap scores
        ref_scores_c = self.proj_score(ref_feats_c)  # (N_c, 1)
        src_scores_c = self.proj_score(src_feats_c)  # (M_c, 1)
        latent_list = [torch.cat([ref_scores_c, src_scores_c], dim=0)]

        if self.cross_score:
            # Each node aggregates the scores of the nodes it is most similar to in the other cloud
            ref_feats_c_norm = F.normalize(ref_feats_c, p=2, dim=1)
            src_feats_c_norm = F.normalize(src_feats_c, p=2, dim=1)
            similarity = torch.matmul(ref_feats_c_norm, src_feats_c_norm.transpose(0, 1))  # (N_c, M_c)
            temperature = torch.exp(self.epsilon) + 0.03
            ref_cross_scores_c = torch.matmul(F.softmax(similarity / temperature, dim=1), src_scores_c)
            src_cross_scores_c = torch.matmul(F.softmax(similarity.transpose(0, 1) / temperature, dim=1), ref_scores_c)
            latent_list.append(torch.cat([ref_cross_scores_c, src_cross_scores_c], dim=0))

        latent_list.append(torch.cat([ref_feats_c, src_feats_c], dim=0))
        latent_c = torch.cat(latent_list, dim=1)

        # 5. Decoder -> per-point overlap logits at the finest level
        logits = self.decoder(latent_c, encoder_feats_list, data_dict)
        probs = torch.sigmoid(logits.detach())

        output_dict['ref_overlap_logits'] = logits[:ref_length]
        output_dict['src_overlap_logits'] = logits[ref_length:]
        output_dict['ref_overlap_probs'] = probs[:ref_length]
        output_dict['src_overlap_probs'] = probs[ref_length:]

        return output_dict


def create_model(config):
    model = GeoPredator(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
