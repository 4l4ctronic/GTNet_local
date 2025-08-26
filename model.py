#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2020/3/9 9:32 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from pointnet_util import index_points
from transformer_divide import TransformerBlock, Attention, GT, SA_Layer,get_graph_feature


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
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
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
def sample_and_group(npoint, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint，3]
    new_points = index_points(points, fps_idx) ## [B, npoint，D]（D=64）
    return  new_xyz,new_points
class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x): #x:[B,C,N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        #F.adaptive_max_pool1d对输入应用一维自适应池
        x = F.adaptive_max_pool1d(x, 1).squeeze() #squeeze去掉维度为1的维度  x:[B,1024]
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

class GTNet_cls(nn.Module):
    def __init__(self, args, output_channels=40):
        super(GTNet_cls, self).__init__()
        self.args = args
        self.k = args.k

        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.bn3 = nn.BatchNorm1d(128)
        # self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.transform_net = Transform_Net(args)

        self.lrgm1 = LRGM(3,   64,  k=self.k, dilation=1, shape_repr="cylinder",
                          use_corr=True, use_residual=True)
        self.lrgm2 = LRGM(64,  64,  k=self.k, dilation=2, shape_repr="cylinder",
                          use_corr=True, use_residual=True)
        self.lrgm3 = LRGM(64,  128, k=self.k, dilation=3, shape_repr="cylinder",
                          use_corr=True, use_residual=True)
        self.lrgm4 = LRGM(128, 256, k=self.k, dilation=4, shape_repr="cylinder",
                          use_corr=True, use_residual=True)

        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))#emb_dims=1024

        # add
"""         self.transformer1 = GT(3, 64,self.k)
        self.transformer2 = GT(64, 64,self.k)
        self.transformer3 = GT(64, 128,self.k)
        self.transformer4 = GT(128, 256,self.k) """
        # //
        self.fc1 = nn.Linear(args.emb_dims,512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=args.dropout)
        self.fc2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=args.dropout)
        self.fc_out=nn.Linear(256,output_channels)
# --- in GTNet_cls.forward ---
    def forward(self, x):                  # x: (B,3,N)
        B, _, N = x.shape
        xyz = x                            # 用坐标做 IDKNN
        feat = x                           # 初始特征 C=3

        idx1 = self.lrgm1.build_idx(xyz)
        f1, _ = self.lrgm1(feat, xyz, idx1)  # (B,64,N)
        idx2 = self.lrgm2.build_idx(f1)
        f2, _ = self.lrgm2(f1,   xyz, idx2)  # (B,64,N)
        idx3 = self.lrgm3.build_idx(f2)
        f3, _ = self.lrgm3(f2,   xyz, idx3)  # (B,128,N)
        idx4 = self.lrgm4.build_idx(f3)
        f4, _ = self.lrgm4(f3,   xyz, idx4)  # (B,256,N)

        x = torch.cat([f1, f2, f3, f4], dim=1)   # (B,512,N)
        x = self.conv5(x)                        # (B,emb_dims,N)  emb_dims=1024

        x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)  # (B,1024)
        x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)  # (B,1024)
        x  = x1 - x2

        x  = F.leaky_relu(self.bn6(self.fc1(x)), 0.2)
        x  = self.dp1(x)
        x  = F.leaky_relu(self.bn7(self.fc2(x)), 0.2)
        x  = self.dp2(x)
        x  = self.fc_out(x)                        # (B,num_classes)
        return x



                                  
class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x) # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        torch.cuda.empty_cache()
        x = self.conv2(x)
        torch.cuda.empty_cache()# (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class GTNet_partseg(nn.Module):
    def __init__(self, args, seg_num_all):
        super(GTNet_partseg, self).__init__()
        self.args = args
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.transform_net = Transform_Net(args)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(64)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        
        self.conv6 = nn.Sequential(nn.Conv1d(96*4+3, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(963, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))

        # add

        self.transformer1 =GT(3, 96,k=self.k)
       
        self.transformer2 = GT(96, 96,k=self.k)

        

        self.transformer3 = GT(96, 96,k=self.k)
        self.transformer4 = GT(96, 96,k=self.k)
       
       

        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0, _ = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)  # (batch_size, 3, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        # x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        # xyz = x
        x = x.transpose(2, 1)
        x_yuan=x
       
        x1 = self.transformer1(x)[0]
       
        x2 = self.transformer2(x1)[0]
       
        x3 = self.transformer3(x2)[0]
        x4 = self.transformer4(x3)[0]
       
        x = torch.cat((x_yuan,x1, x2, x3,x4), dim=1)
        # x = x.permute(0, 2, 1)
        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
       
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
       
        l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1) num_categoties=16（包含类型）
        l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)
        # x = x.permute(0, 2, 1)
        x = torch.cat((x_yuan,x, x1, x2, x3,x4), dim=1)  # (batch_size, 1088+64*3, num_points)
        # x = x.permute(0, 2, 1)
        # add

        # //
        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
       
        # add
        x = self.dp1(x)
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        # x=self.self_attn6(x) #add
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        return x



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
def chazhi(xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """

        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:

            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        return new_points



class GTNet_semseg(nn.Module):
    def __init__(self, args):
        super(GTNet_semseg, self).__init__()
        self.args = args
        self.k = args.k

       
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn10=nn.BatchNorm1d(128)
        self.bn9=nn.BatchNorm1d(13)
        
        self.conv6 = nn.Sequential(nn.Conv1d(384, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1024+384, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 =nn.Conv1d(256, 13, kernel_size=1, bias=False)
        # self.conv10=nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
        #                            self.bn8,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv11=nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                           
                                   nn.LeakyReLU(negative_slope=0.2))
       
        self.transformer1 = GT(9, 96,self.k)
        self.transformer2 = GT(96,96,self.k)
        self.transformer3 = GT(96,96,self.k)
        self.transformer4=GT(96,96,self.k)
    def forward(self, x):
        x=x.permute(0,2,1)
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # x = x.permute(0, 2, 1)
        x1 = self.transformer1(x,dim9=True)[0]
       
        x2 = self.transformer2(x1)[0]
       
        x3 = self.transformer3(x2)[0]
        x4=self.transformer4(x3)[0]
        

        x = torch.cat((x1, x2, x3,x4), dim=2).permute(0,2,1)      # (batch_size, 64*3, num_points)
       
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        avgs = F.adaptive_avg_pool1d(x,1).view(batch_size,-1,1)      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        maxs=F.adaptive_max_pool1d(x,1).view(batch_size,-1,1)
        x=maxs-avgs
        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x=x.permute(0,2,1)
        x = torch.cat((x, x1,x2,x3,x4), dim=2)   # (batch_size, 1024+64*3, num_points)
        x = x.permute(0, 2, 1)
        # add
        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
       
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        torch.cuda.empty_cache()
        # x = self.conv9(x)
        return x

class LRGM(nn.Module):
    def __init__(self, in_channels, out_channels, k=16, dilation=1,
                 shape_repr="cylinder", use_corr=True, use_residual=True,
                 dropout=0.0, act_layer=nn.ReLU, norm="bn"):
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.shape_repr = shape_repr
        self.use_corr = use_corr
        self.use_residual = use_residual

        shape_c = 13
        in_c2d = in_channels + shape_c

        self.mlp = nn.Sequential(
            nn.Conv2d(in_c2d, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels) if norm == "bn" else nn.Identity(),
            act_layer(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        ) 

        if use_residual:
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm1d(out_channels) if norm == "bn" else nn.Identity(),
                )
            else:
                self.shortcut = nn.Identity()
            self.act = act_layer(inplace=True)

        # === LFCM learnable mapping along K ===
        self.wK = nn.Linear(k, k, bias=False) if use_corr else None
        #wK 初始化为恒等（训练更稳）
        if self.wK is not None:
            nn.init.eye_(self.wK.weight)

    @torch.no_grad()
    def build_idx(self, base):
        idx, _ = knn_with_dilation(base, self.k, self.dilation)  # xyz: (B,3,N)
        return idx

    def forward(self, x, xyz, idx=None):
        """
        x:   (B,C,N)
        xyz: (B,3,N)
        idx: (B,N,K)
        """
        B, C, N = x.shape
        if idx is None:
            idx = self.build_idx(xyz)           # 用坐标做 KNN



        # 邻居特征与中心特征
        x_j = gather_feat_neighbors(x, idx)                 # (B,C,N,K)
        x_i = x.unsqueeze(-1).expand(-1, -1, -1, self.k)    # (B,C,N,K)
        feat_rel = x_j - x_i                                 # ΔF

        # LPR 13D
        xyz_bnc, knn_xyz = gather_xyz_neighbors(xyz, idx)   # (B,N,3), (B,N,K,3)
        if self.shape_repr == "cylinder":
            lshape = LocalFeatureRepresentaion_cylinder(xyz_bnc, knn_xyz, nsample=self.k)  # (B,N,K,13)
        else:
            lshape, _, _ = LocalFeatureRepresentaion_polar(xyz_bnc, knn_xyz, return_dis=True)  # (B,N,K,13)
        lshape = lshape.permute(0, 3, 1, 2).contiguous()     # (B,13,N,K)

        # LFCM: 可学习的 K 维注意力
        if self.use_corr:
            xi = F.normalize(x_i, dim=1)
            xj = F.normalize(x_j, dim=1)
            corr = (xi * xj).sum(1, keepdim=True)           # (B,1,N,K)
            a = corr.squeeze(1).reshape(B * N, self.k)      # (B*N, K)
            a = self.wK(a)
            a = torch.softmax(a, dim=-1).view(B, 1, N, self.k)
            feat_rel = feat_rel * a + feat_rel              # 论文式(2)

        # 共享 MLP + MAX
        f = torch.cat([feat_rel, lshape], dim=1)             # (B,C+13,N,K)
        f = self.mlp(f)                                      # (B,Co,N,K)
        f = f.max(dim=-1, keepdim=False)[0]                  # (B,Co,N)

        # 残差
        if self.use_residual:
            f = self.act(f + self.shortcut(x))

        return f, idx


# === 聚合包装函数 ===
def gather_feat_neighbors(x, idx):
    x_bnc = x.transpose(1, 2)                # (B,N,C)
    x_nbr = index_points(x_bnc, idx)         # (B,N,K,C)
    return x_nbr.permute(0, 3, 1, 2).contiguous()  # (B,C,N,K)

def gather_xyz_neighbors(xyz, idx):
    xyz_bnc = xyz.transpose(1, 2)            # (B,N,3)
    knn_xyz = index_points(xyz_bnc, idx)     # (B,N,K,3)
    return xyz_bnc, knn_xyz

def index_points(points, idx, cuda=False, is_group=False):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

#3个方位的极坐标的表达方式的叠加
def  LocalFeatureRepresentaion_polar(xyz,knn_points,return_dis=True):
    # xyz:b,n,3
    # knn_points:b,n,k,3
    b,n,k,_ = knn_points.shape
    eps=1e-9
    knn_points_norm = knn_points - xyz.unsqueeze(-2) # b,n,k,3 去心之后,相对位置
    local_dis = torch.sqrt(torch.sum(knn_points_norm **2 ,dim=-1)+eps) # b,n,k
    local_x = knn_points_norm[:,:,:,0] # b,n,k
    local_y = knn_points_norm[:,:,:,1]# b,n,k
    local_z = knn_points_norm[:,:,:,2] # b,n,k
    local_xy = torch.sqrt(local_x ** 2 + local_y ** 2+eps)  # b,n, k
    local_xz = torch.sqrt(local_x ** 2 + local_z ** 2+eps) # b,n.k
    local_yz = torch.sqrt(local_y ** 2 + local_z ** 2+eps) # b,n,k

    # center_mass = torch.mean(knn_points_norm,dim=-2)  # b,n,3
    # z_fi_center = torch.atan2(center_mass[:,:,1], center_mass[:,:,0]) # b,n
    # z_ceta_center = torch.atan2(center_mass[:,:,2], torch.sqrt(center_mass[:,:,0] ** 2 + center_mass[:,:,1] ** 2))
    #
    # y_fi_center = torch.atan2(center_mass[:,:,0],center_mass[:,:,2])
    # y_ceta_center = torch.atan2(center_mass[:,:,1],torch.sqrt(center_mass[:,:,0] ** 2 + center_mass[:,:,2] ** 2))
    #
    # x_fi_center = torch.atan2(center_mass[:,:,2],center_mass[:,:,1])
    # x_ceta_center = torch.atan2(center_mass[:,:,0], torch.sqrt(center_mass[:,:,1] ** 2 + center_mass[:,:,2] ** 2))

    # z- invariant features
    z_fi = torch.atan2(local_y,local_x) # b,n,k
    z_ceta = torch.atan2(local_z,local_xy) # b,n,k
    # z_fi = z_fi - z_fi_center.unsqueeze(-1)
    # z_ceta = z_ceta - z_ceta_center.unsqueeze(-1)

    # y-invariant
    y_fi = torch.atan2(local_z,local_x) # b,n,k
    y_ceta = torch.atan2(local_y,local_xz) # b,n,k
    # y_fi = y_fi - y_fi_center.unsqueeze(-1)
    # y_ceta = y_ceta - y_ceta_center.unsqueeze(-1)

    #x-invariant
    x_fi = torch.atan2(local_z,local_y) #b,n,k
    x_ceta = torch.atan2(local_x,local_yz) # b, n, k
    # x_fi = x_fi - x_fi_center.unsqueeze(-1)
    # x_ceta = x_ceta - x_ceta_center.unsqueeze(-1)
    #source pos
    xyzlift = xyz.unsqueeze(-2).repeat(1,1,k,1)  # ,b,n,k,3
    #taerget pos   knn_points  absolute pos
    if return_dis:
        local_feature = torch.cat((local_dis.unsqueeze(-1),z_fi.unsqueeze(-1),z_ceta.unsqueeze(-1),y_fi.unsqueeze(-1)
                               ,y_ceta.unsqueeze(-1),x_fi.unsqueeze(-1),x_ceta.unsqueeze(-1),knn_points_norm,xyzlift),dim = -1)   # b,n,k,10+3
    else:
        local_feature = torch.cat(( z_fi.unsqueeze(-1), z_ceta.unsqueeze(-1), y_fi.unsqueeze(-1)  ,  y_ceta.unsqueeze(-1),x_fi.unsqueeze(-1),  x_ceta.unsqueeze(-1)
                                   ,  knn_points_norm,
                                   xyzlift), dim=-1)  # b,n,k,12


    return local_feature,local_dis.unsqueeze(-1),knn_points_norm # b,n,k,1

def LocalFeatureRepresentaion_cylinder(xyz,knn_points,nsample,dila3 = False):
    """

    :param xyz: b,n,3
    :param knn_points: b,n,k,3
    :param nsample: k
    :param radius: r
    :param ball: use knn or ball query
    :return:
    """

    # knn_points = index_points(xyz,knn_index) # b,n,k,3
    knn_points_norm = knn_points - xyz.unsqueeze(-2) # b,n,k,3 去心之后,相对位置
    eps=1e-9
    local_dis = torch.sqrt(torch.sum(knn_points_norm **2 ,dim=-1)+eps) # b,n,k

    # center_mass = torch.mean(knn_points_norm,dim = -2)# b,n,3
    # z_ceta_center = torch.atan2(center_mass[:,:,1],center_mass[:,:,0]) # b,n
    # y_ceta_center = torch.atan2(center_mass[:,:,0],center_mass[:,:,2])
    # x_ceta_center = torch.atan2(center_mass[:,:,2],center_mass[:,:,1])

    #z_invarient
    z_z = knn_points_norm[:,:,:,2] # b,n,k
    z_r = torch.sqrt(knn_points_norm[:,:,:,0] ** 2 + knn_points_norm[:,:,:,1] ** 2+eps) # b,n,k
    z_ceta = torch.atan2(knn_points_norm[:,:,:,1],knn_points_norm[:,:,:,0])
    # z_ceta = z_ceta - z_ceta_center.unsqueeze(-1) # b,n,k

    # y-invariant
    y_y = knn_points_norm[:,:,:,1] # b,n,k
    y_r = torch.sqrt(knn_points_norm[:,:,:,0] ** 2 + knn_points_norm[:,:,:,2] ** 2+eps)
    y_ceta = torch.atan2(knn_points_norm[:,:,:,0],knn_points_norm[:,:,:,2])
    # y_ceta = y_ceta  - y_ceta_center.unsqueeze(-1) # b,n,k

    # x_invariant
    x_x = knn_points_norm[:,:,:,0] # b,n,k
    x_r = torch.sqrt(knn_points_norm[:,:,:,1] ** 2 + knn_points_norm[:,:,:,2] ** 2+eps)
    x_ceta = torch.atan2(knn_points_norm[:,:,:,2],knn_points_norm[:,:,:,1])
    # x_ceta = x_ceta - x_ceta_center.unsqueeze(-1)  # b,n,k
    if dila3:
        xyz_lift = xyz.unsqueeze(-2).repeat(1,1,3*nsample,1)  # b,n,k,3
    else:
        xyz_lift = xyz.unsqueeze(-2).repeat(1, 1, nsample, 1)  # b,n,k,3

    local_features = torch.cat((z_z.unsqueeze(-1),y_y.unsqueeze(-1),x_x.unsqueeze(-1),z_r.unsqueeze(-1),z_ceta.unsqueeze(-1),y_r.unsqueeze(-1),y_ceta.unsqueeze(-1)
                                ,x_r.unsqueeze(-1),x_ceta.unsqueeze(-1),local_dis.unsqueeze(-1),xyz_lift),dim = -1)  # b,n,k,10+3

    return local_features

def knn_with_dilation(x, k, d):
    """

    :param x:  b,c,n 可以是点的位置，也可以是点的特征
    :param k:  k nearest  points
    :param d:  dilation
    :return:   b, n, k :index
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # b,n,n
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # b, 1, n
    pair_distance = -xx - inner - xx.transpose(2, 1)  # b, n,n  ascending
    w = k // d
    assert w > 1
    feature_distance = pair_distance.topk(k=k * d, dim=-1)[0]  # b,n,k*d
    idxall = pair_distance.topk(k=k * d, dim=-1)[1]  # b, n, k*d
    if d == 1:  # if w > 1, need unsqueeze(-1) to keep dim
        retidx = idxall[:, :, :k]  # b,n,k
        retdis = feature_distance[:, :, :k]
    elif d == 2:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w + 2 * (k - w) + 1:2]), dim=-1)  # b, n,k
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w + 2 * (k - w) + 1:2]), dim=-1)  # b, n,k
    elif d == 3:
        retidx = torch.cat(
            (idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]),
            dim=-1)
        retdis = torch.cat(
            (feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2], feature_distance[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]),
            dim=-1)
    elif d == 4:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2], feature_distance[:, :, 3 * w + 2: 6 * w:3],
                            feature_distance[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
    elif d == 5:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2], idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3: w * 10:4], idxall[:, :, 10 * w + 4:5 * (k - 4 * w) + 10 * w + 1:5]),
                           dim=-1)
        retdis = torch.cat((feature_distance[:, :, :w], feature_distance[:, :, w + 1:w * 3:2], feature_distance[:, :, 3 * w + 2: 6 * w:3],
                            feature_distance[:, :, 6 * w + 3: w * 10:4], feature_distance[:, :, 10 * w + 4:5 * (k - 4 * w) + 10 * w + 1:5]),
                           dim=-1)
    return retidx,retdis

""" B, N, C0 = 2, 1024, 8
C1 = 64
xyz = torch.randn(B, 3, N)
x0  = torch.randn(B, C0, N)

lrgm = LRGM(in_channels=C0, out_channels=C1, k=16, dilation=2,
            shape_repr="cylinder", use_corr=True, use_residual=True)

with torch.no_grad():
    f1, idx1 = lrgm(x0, xyz, idx=None)
print(f1.shape, idx1.shape)  # torch.Size([B, C1, N])  torch.Size([B, N, 16]) """
