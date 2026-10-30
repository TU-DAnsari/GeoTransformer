import numpy as np

from dataset_geoseg import GeoSegDataset

from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def object_scene_collate_fn_stack_mode(
    data_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=True
):
    r"""Collate function for object segmentation in stack mode.

    Each sample provides `points_scene` (N, 3), `points_object` (M, 3) and per-scene-point binary `labels` (N,).
    The scene is stacked first and the object second: [scene_1, ..., scene_B, object_1, ..., object_B].
    Input features are all ones. The first stage is not subsampled, so `labels` stay aligned with `points[0]`.
    """
    stack_dicts = []
    for data_dict in data_dicts:
        stack_dict = {key: value for key, value in data_dict.items() if key not in ['scene_points', 'objects_points']}
        scene_points = np.asarray(data_dict['points_scene'], dtype=np.float32)
        object_points = np.asarray(data_dict['points_object'], dtype=np.float32)
        stack_dict['ref_points'] = scene_points
        stack_dict['src_points'] = object_points
        stack_dict['ref_feats'] = np.ones((scene_points.shape[0], 1), dtype=np.float32)
        stack_dict['src_feats'] = np.ones((object_points.shape[0], 1), dtype=np.float32)
        stack_dict['labels'] = np.asarray(data_dict['labels']).astype(bool)
        stack_dicts.append(stack_dict)

    return registration_collate_fn_stack_mode(
        stack_dicts, num_stages, voxel_size, search_radius, neighbor_limits, precompute_data=precompute_data
    )


def train_valid_data_loader(cfg, distributed):
    train_dataset = GeoSegDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        object_scene_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        object_scene_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = GeoSegDataset(
        cfg.data.dataset_root,
        'val',
        point_limit=cfg.test.point_limit,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        object_scene_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg, benchmark):
    train_dataset = GeoSegDataset(
        cfg.data.dataset_root,
        'train',
        point_limit=cfg.train.point_limit,
    )
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        object_scene_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    test_dataset = GeoSegDataset(
        cfg.data.dataset_root,
        benchmark,
        point_limit=cfg.test.point_limit,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        object_scene_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
    )

    return test_loader, neighbor_limits
