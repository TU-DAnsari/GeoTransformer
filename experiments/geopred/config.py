import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from geotransformer.utils.common import ensure_dir


_C = edict()

# common
_C.seed = 7351

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))
_C.exp_name = osp.basename(_C.working_dir)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')
_C.feature_dir = osp.join(_C.output_dir, 'features')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)

# data
_C.data = edict()
_C.data.dataset_root = osp.join("/home/danish/lobster/ml/data", '3DMatch')

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 2
_C.train.point_limit = 30000
_C.train.use_augmentation = True
_C.train.augmentation_noise = 0.005
_C.train.augmentation_rotation = 1.0

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 2
_C.test.point_limit = 30000

# evaluation
_C.eval = edict()
_C.eval.overlap_threshold = 0.5

# optim
_C.optim = edict()
_C.optim.lr = 1e-4
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 40
_C.optim.grad_acc_steps = 1

# model - backbone
_C.backbone = edict()
_C.backbone.num_stages = 4
_C.backbone.init_voxel_size = 0.025
_C.backbone.kernel_size = 15
_C.backbone.base_radius = 2.5
_C.backbone.base_sigma = 2.0
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size
_C.backbone.group_norm = 32
_C.backbone.input_dim = 1
_C.backbone.init_dim = 64

# model - Global
_C.model = edict()
_C.model.overlap_radius = 0.0375  # Predator: 1.5 x voxel size

# model - GeoTransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 1024
_C.geotransformer.hidden_dim = 256
_C.geotransformer.output_dim = 256
_C.geotransformer.num_heads = 4
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
_C.geotransformer.sigma_d = 0.2
_C.geotransformer.sigma_a = 15
_C.geotransformer.angle_k = 3
_C.geotransformer.reduction_a = 'max'

# model - Overlap head
_C.overlap = edict()
_C.overlap.cross_score = True  # Predator's "add_cross_score"

# loss - Overall
_C.loss = edict()
_C.loss.weight_overlap_loss = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')


if __name__ == '__main__':
    main()
