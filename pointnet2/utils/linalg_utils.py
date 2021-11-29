from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from enum import Enum
import numpy as np


PDist2Order = Enum("PDist2Order", "d_first d_second")


def pdist2(X, Z=None, order=PDist2Order.d_second):
    # type: (torch.Tensor, torch.Tensor, PDist2Order) -> torch.Tensor
    r""" Calculates the pairwise distance between X and Z

    D[b, i, j] = l2 distance X[b, i] and Z[b, j]

    Parameters
    ---------
    X : torch.Tensor
        X is a (B, N, d) tensor.  There are B batches, and N vectors of dimension d
    Z: torch.Tensor
        Z is a (B, M, d) tensor.  If Z is None, then Z = X

    Returns
    -------
    torch.Tensor
        Distance matrix is size (B, N, M)
    """

    if order == PDist2Order.d_second:
        if X.dim() == 2:
            X = X.unsqueeze(0)
        if Z is None:
            Z = X
            G = np.matmul(X, Z.transpose(-2, -1))
            S = (X * X).sum(-1, keepdim=True)
            R = S.transpose(-2, -1)
        else:
            if Z.dim() == 2:
                Z = Z.unsqueeze(0)
            G = np.matmul(X, Z.transpose(-2, -1))
            S = (X * X).sum(-1, keepdim=True)
            R = (Z * Z).sum(-1, keepdim=True).transpose(-2, -1)
    else:
        if X.dim() == 2:
            X = X.unsqueeze(0)
        if Z is None:
            Z = X
            G = np.matmul(X.transpose(-2, -1), Z)
            R = (X * X).sum(-2, keepdim=True)
            S = R.transpose(-2, -1)
        else:
            if Z.dim() == 2:
                Z = Z.unsqueeze(0)
            G = np.matmul(X.transpose(-2, -1), Z)
            S = (X * X).sum(-2, keepdim=True).transpose(-2, -1)
            R = (Z * Z).sum(-2, keepdim=True)

    return torch.abs(R + S - 2 * G).squeeze(0)

def pdist2_slow(X, Z=None):
    if Z is None:
        Z = X
    D = torch.zeros(X.size(0), X.size(2), Z.size(2))

    for b in range(D.size(0)):
        for i in range(D.size(1)):
            for j in range(D.size(2)):
                D[b, i, j] = torch.dist(X[b, :, i], Z[b, :, j])
    return D



def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# def farthest_point_sample(xyz, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [B, N, C]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [B, npoint]
#     """
#     device = xyz.device
#     B, N, C = xyz.shape
#     centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
#     distance = torch.ones(B, N).to(device) * 1e10
#     farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
#     batch_indices = torch.arange(B, dtype=torch.long).to(device)
#
#     for i in range(npoint):
#         centroids[:, i] = farthest
#         centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
#         dist = torch.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = torch.max(distance, -1)[1]
#
#     return centroids

def FarthestPointSample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    dist = square_distance(xyz, xyz)
    for i in range(npoint):
        centroids[:, i] = farthest
        cur_seed_d = dist[batch_indices, farthest, :] #[B, N]

        mask = cur_seed_d < distance
        distance[mask] = cur_seed_d[mask]

        farthest = torch.argmax(distance, -1)
    return centroids

if __name__ == "__main__":

    X = torch.randn(1, 5, 3)
    Z = torch.randn(2, 3, 3)

    idx = FarthestPointSample(X, 3)

    print(idx)

    # print(pdist2(X, order=PDist2Order.d_first))
    # print(pdist2_slow(X))
    # print(torch.dist(pdist2(X, order=PDist2Order.d_first), pdist2_slow(X)))
