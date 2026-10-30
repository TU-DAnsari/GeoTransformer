import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.geotransformer import GeometricTransformer

from backbone import KPConvEncoder, OverlapDecoder


class GeoObjSeg(nn.Module):
    r"""KPConv encoder -> GeoTransformer self/cross attention between scene and object -> Predator-style head.

    The only output is the per-point probability that a scene point belongs to the object. No transformation or
    ground truth is computed here: the labels come straight from the dataset.
    """

    def __init__(self, cfg):
        super(GeoObjSeg, self).__init__()
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

        # Predator: coarse score from the conditioned features, optionally cross-attended to the other cloud
        self.proj_score = nn.Linear(cfg.geotransformer.output_dim, 1)
        if self.cross_score:
            self.epsilon = nn.Parameter(torch.tensor(-5.0))

        decoder_input_dim = cfg.geotransformer.output_dim + (2 if self.cross_score else 1)
        self.decoder = OverlapDecoder(decoder_input_dim, cfg.backbone.init_dim, cfg.backbone.group_norm)

    def forward(self, data_dict):
        output_dict = {}

        feats = data_dict['features'].detach()

        # scene is stacked first, object second (see object_scene_collate_fn_stack_mode)
        scene_length_c = data_dict['lengths'][-1][0].item()
        scene_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points = data_dict['points'][0].detach()

        scene_points_c = points_c[:scene_length_c]
        object_points_c = points_c[scene_length_c:]
        scene_points = points[:scene_length]
        object_points = points[scene_length:]

        output_dict['scene_points_c'] = scene_points_c
        output_dict['object_points_c'] = object_points_c
        output_dict['scene_points'] = scene_points
        output_dict['object_points'] = object_points

        # 1. KPConv Encoder
        encoder_feats_list = self.encoder(feats, data_dict)
        feats_c = encoder_feats_list[-1]

        # 2. Geometric Transformer (self + cross attention)
        scene_feats_c = feats_c[:scene_length_c]
        object_feats_c = feats_c[scene_length_c:]
        scene_feats_c, object_feats_c = self.transformer(
            scene_points_c.unsqueeze(0),
            object_points_c.unsqueeze(0),
            scene_feats_c.unsqueeze(0),
            object_feats_c.unsqueeze(0),
        )
        scene_feats_c = scene_feats_c.squeeze(0)
        object_feats_c = object_feats_c.squeeze(0)

        # 3. Coarse scores
        scene_scores_c = self.proj_score(scene_feats_c)  # (N_c, 1)
        object_scores_c = self.proj_score(object_feats_c)  # (M_c, 1)
        latent_list = [torch.cat([scene_scores_c, object_scores_c], dim=0)]

        if self.cross_score:
            # Each node aggregates the scores of the nodes it is most similar to in the other cloud
            scene_feats_c_norm = F.normalize(scene_feats_c, p=2, dim=1)
            object_feats_c_norm = F.normalize(object_feats_c, p=2, dim=1)
            similarity = torch.matmul(scene_feats_c_norm, object_feats_c_norm.transpose(0, 1))  # (N_c, M_c)
            temperature = torch.exp(self.epsilon) + 0.03
            scene_cross_scores_c = torch.matmul(F.softmax(similarity / temperature, dim=1), object_scores_c)
            object_cross_scores_c = torch.matmul(
                F.softmax(similarity.transpose(0, 1) / temperature, dim=1), scene_scores_c
            )
            latent_list.append(torch.cat([scene_cross_scores_c, object_cross_scores_c], dim=0))

        latent_list.append(torch.cat([scene_feats_c, object_feats_c], dim=0))
        latent_c = torch.cat(latent_list, dim=1)

        # 4. Decoder -> per-point logits at the finest level; only the scene half is supervised
        logits = self.decoder(latent_c, encoder_feats_list, data_dict)
        scene_logits = logits[:scene_length]

        output_dict['scene_logits'] = scene_logits
        output_dict['scene_probs'] = torch.sigmoid(scene_logits.detach())

        return output_dict


def create_model(config):
    model = GeoObjSeg(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()
