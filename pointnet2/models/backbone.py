from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2.utils.pointnet2_modules import PointnetSAModule, PointnetFPModule, PointnetSAModuleVotes


# PointNet++
class PointNet2(nn.Module):
    """get point-wise feature"""

    def __init__(self, input_channels=3, use_xyz=True):
        super(PointNet2, self).__init__()

        skip_channel_list = [input_channels, 128, 256, 256]

        self.SA_module = nn.ModuleList()
        self.SA_module.append(
            PointnetSAModule(mlp=[input_channels, 64, 64, 128],
                             radius=0.3,
                             nsample=32,
                             bn=True,
                             use_xyz=use_xyz)
        )
        self.SA_module.append(
            PointnetSAModule(mlp=[128, 128, 128, 256],
                             radius=0.5,
                             nsample=32,
                             bn=True,
                             use_xyz=use_xyz)
        )
        self.SA_module.append(
            PointnetSAModule(mlp=[256, 256, 256, 256],
                             radius=0.7,
                             nsample=32,
                             bn=True,
                             use_xyz=use_xyz)
        )

        self.trans = nn.Conv1d(256, 256, kernel_size=1)

        self.FP_module = nn.ModuleList()
        self.FP_module.append(
            PointnetFPModule(mlp=[256 + skip_channel_list[-2], 128, 128], bn=True)
        )
        self.FP_module.append(
            PointnetFPModule(mlp=[128 + skip_channel_list[-3], 256, 256], bn=True)
        )
        self.FP_module.append(
            PointnetFPModule(mlp=[256 + skip_channel_list[-4], 256, 256], bn=True)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, numpoints):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_module)):
            li_xyz, li_features = self.SA_module[i](l_xyz[i], l_features[i], numpoints[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        mid_feat = self.trans(l_features[-1])
        mid_xyz = l_xyz[-1]
        for i in range(len(self.FP_module)):
            j = -(i + 1)
            l_features[j - 1] = self.FP_module[i](l_xyz[j - 1], l_xyz[j], l_features[j - 1], l_features[j])

        return l_xyz[0], l_features[0], mid_xyz, mid_feat  # [B, N, 3], [B, C, N]


class PointNet2Ind(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network.

       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """

    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.3,
            nsample=32,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=256,
            radius=0.5,
            nsample=32,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=128,
            radius=0.7,
            nsample=32,
            mlp=[256, 256, 256, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.trans = nn.Conv1d(256, 256, kernel_size=1)

        self.fp1 = PointnetFPModule(mlp=[256 + 256, 128, 128])
        self.fp2 = PointnetFPModule(mlp=[128 + 128, 256, 256])
        self.fp3 = PointnetFPModule(mlp=[256 + input_feature_dim, 256, 256])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        end_points['in_xyz'] = xyz
        end_points['in_features'] = features

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features)  # this fps_inds is just 0,1,...,255
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features)  # this fps_inds is just 0,1,...,127
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        end_points['mid_features'] = self.trans(features)

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa2_xyz'], end_points['sa3_xyz'],
                            end_points['sa2_features'], end_points['sa3_features'])
        features = self.fp2(end_points['sa1_xyz'], end_points['sa2_xyz'],
                            end_points['sa1_features'], features)
        features = self.fp3(end_points['in_xyz'], end_points['sa1_xyz'],
                            end_points['in_features'], features)
        end_points['fp3_features'] = features

        num_seed = end_points['sa3_xyz'].shape[1]
        end_points['seed_inds'] = end_points['sa1_inds'][:, 0:num_seed]  # indices among the entire input point clouds
        return end_points['in_xyz'], end_points['fp3_features'], end_points['sa3_xyz'], end_points['mid_features'], end_points['seed_inds']


# DGCNN
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20):
    # x = x.squeeze()
    x = x.view(*x.size()[:3])
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)

    return feature

class DGCNN(nn.Module):
    def __init__(self, n_emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(n_emb_dims)

    def forward(self, x):
        x_ = x.clone()
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x3)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2).view(batch_size, -1, num_points)
        return x_, x
