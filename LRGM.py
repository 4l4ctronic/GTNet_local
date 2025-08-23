import torch
import torch.nn as nn
import torch.nn.functional as F

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

    @torch.no_grad()
    def build_idx(self, xyz):
        idx, _ = knn_with_dilation(xyz, self.k, self.dilation)  # xyz: (B,3,N)
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
    knn_points_norm = knn_points - xyz.unsqueeze(-2) # b,n,k,3 去心之后,相对位置
    local_dis = torch.sqrt(torch.sum(knn_points_norm **2 ,dim=-1)) # b,n,k
    local_x = knn_points_norm[:,:,:,0] # b,n,k
    local_y = knn_points_norm[:,:,:,1]# b,n,k
    local_z = knn_points_norm[:,:,:,2] # b,n,k
    local_xy = torch.sqrt(local_x ** 2 + local_y ** 2)  # b,n, k
    local_xz = torch.sqrt(local_x ** 2 + local_z ** 2) # b,n.k
    local_yz = torch.sqrt(local_y ** 2 + local_z ** 2) # b,n,k

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
    local_dis = torch.sqrt(torch.sum(knn_points_norm **2 ,dim=-1)) # b,n,k

    # center_mass = torch.mean(knn_points_norm,dim = -2)# b,n,3
    # z_ceta_center = torch.atan2(center_mass[:,:,1],center_mass[:,:,0]) # b,n
    # y_ceta_center = torch.atan2(center_mass[:,:,0],center_mass[:,:,2])
    # x_ceta_center = torch.atan2(center_mass[:,:,2],center_mass[:,:,1])

    #z_invarient
    z_z = knn_points_norm[:,:,:,2] # b,n,k
    z_r = torch.sqrt(knn_points_norm[:,:,:,0] ** 2 + knn_points_norm[:,:,:,1] ** 2) # b,n,k
    z_ceta = torch.atan2(knn_points_norm[:,:,:,1],knn_points_norm[:,:,:,0])
    # z_ceta = z_ceta - z_ceta_center.unsqueeze(-1) # b,n,k

    # y-invariant
    y_y = knn_points_norm[:,:,:,1] # b,n,k
    y_r = torch.sqrt(knn_points_norm[:,:,:,0] ** 2 + knn_points_norm[:,:,:,2] ** 2)
    y_ceta = torch.atan2(knn_points_norm[:,:,:,0],knn_points_norm[:,:,:,2])
    # y_ceta = y_ceta  - y_ceta_center.unsqueeze(-1) # b,n,k

    # x_invariant
    x_x = knn_points_norm[:,:,:,0] # b,n,k
    x_r = torch.sqrt(knn_points_norm[:,:,:,1] ** 2 + knn_points_norm[:,:,:,2] ** 2)
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

B, N, C0 = 2, 1024, 8
C1 = 64
xyz = torch.randn(B, 3, N)
x0  = torch.randn(B, C0, N)

lrgm = LRGM(in_channels=C0, out_channels=C1, k=16, dilation=2,
            shape_repr="cylinder", use_corr=True, use_residual=True)

with torch.no_grad():
    f1, idx1 = lrgm(x0, xyz, idx=None)
print(f1.shape, idx1.shape)  # torch.Size([B, C1, N])  torch.Size([B, N, 16])