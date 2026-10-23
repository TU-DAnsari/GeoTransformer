import torch.nn as nn

from geotransformer.modules.kpconv import ConvBlock, ResidualBlock, LastUnaryBlock


class KPConvBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        r"""Single-resolution KPConv feature extractor.

        Every block convolves the input points onto themselves with the same radius, so there is no
        subsampling and the output features are aligned 1:1 with the input points. The receptive field
        grows by about one conv radius per block.
        """
        super(KPConvBackbone, self).__init__()

        self.encoder1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder3 = ResidualBlock(init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder4 = ResidualBlock(init_dim * 2, init_dim * 4, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder5 = ResidualBlock(init_dim * 4, init_dim * 4, kernel_size, init_radius, init_sigma, group_norm)

        self.head = LastUnaryBlock(init_dim * 4, output_dim)

    def forward(self, feats, data_dict):
        points = data_dict['points'][0]
        neighbors = data_dict['neighbors'][0]

        feats = self.encoder1(feats, points, points, neighbors)
        feats = self.encoder2(feats, points, points, neighbors)
        feats = self.encoder3(feats, points, points, neighbors)
        feats = self.encoder4(feats, points, points, neighbors)
        feats = self.encoder5(feats, points, points, neighbors)

        feats = self.head(feats)

        return feats
