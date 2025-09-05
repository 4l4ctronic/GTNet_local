import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------------------- helpers (保持你项目里的签名不变) --------------------
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

def gather_feat_neighbors(x, idx):
    x_bnc = x.transpose(1, 2)                # (B,N,C)
    x_nbr = index_points(x_bnc, idx)         # (B,N,K,C)
    return x_nbr.permute(0, 3, 1, 2).contiguous()  # (B,C,N,K)

def gather_xyz_neighbors(xyz, idx):
    xyz_bnc = xyz.transpose(1, 2)            # (B,N,3)
    knn_xyz = index_points(xyz_bnc, idx)     # (B,N,K,3)
    return xyz_bnc, knn_xyz

# -------------------- AttnPool-K：替换 MAX 的轻量注意力聚合 --------------------
class AttnPoolK(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.1, norm="bn"):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c) if norm == "bn" else nn.Identity(),
            nn.GELU(),
        )
        self.score = nn.Conv2d(out_c, 1, 1)  # 每个邻居一个标量分数
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, F):                     # (B,in_c,N,K)
        H = self.phi(F)                       # (B,Co,N,K)
        w = torch.softmax(self.score(H), dim=-1)   # (B,1,N,K)
        H = self.drop(H)
        out = (H * w).sum(dim=-1)            # (B,Co,N)
        return out

# -------------------- LAN-IDKNN：可学习各向异性度量建图 --------------------
class AnisoMetric(nn.Module):
    """为每个中心点预测 diag(sx,sy,sz) 与融合系数 beta（用于特征距离）"""
    def __init__(self, in_c, use_feat=True):
        super().__init__()
        self.use_feat = use_feat
        self.enc = nn.Sequential(
            nn.Conv1d(in_c, 64, 1), nn.ReLU(True),
            nn.Conv1d(64, 32, 1),  nn.ReLU(True),
            nn.Conv1d(32, 3, 1)    # -> (B,3,N)
        )
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_or_xyz):             # (B,C_or_3,N)
        s = F.softplus(self.enc(x_or_xyz)) + 1e-3  # (B,3,N) 保正
        beta = F.softplus(self.beta)
        return s, beta

def _metric_neg_sqdist_xyz(xyz, s):          # (B,3,N),(B,3,N)->(B,N,N)
    # D_ij = sum_c s_{c,i} (x_{c,i}-x_{c,j})^2 ，返回其负值（越大越近）
    term1 = (s * (xyz**2)).sum(1, keepdim=True)        # (B,1,N)
    term2 = -2 * ((s * xyz).transpose(2,1) @ xyz)      # (B,N,N)
    term3 = s.transpose(2,1) @ (xyz**2)                # (B,N,N)
    D = term1.transpose(1,2) + term2 + term3           # (B,N,N)
    return -D

def _neg_sqdist_feat(feat):                   # (B,C,N)->(B,N,N)
    inner = -2 * (feat.transpose(2,1) @ feat)
    ff = (feat**2).sum(1, keepdim=True)
    return -ff - inner - ff.transpose(1,2)

# -------------------- LRGM（集成 AttnPool-K + LAN-IDKNN，其它保持一致） --------------------
class LRGM(nn.Module):
    def __init__(self, in_channels, out_channels, k=16, dilation=1,
                 shape_repr="cylinder", use_corr=True, use_residual=True,
                 dropout=0.1, act_layer=nn.ReLU, norm="bn",
                 use_attnpool=True, use_metric_knn=True,use_repsurf = True):
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.shape_repr = shape_repr
        self.use_corr = use_corr
        self.use_residual = use_residual
        self.use_attnpool = use_attnpool
        self.use_metric_knn = use_metric_knn
        self.use_repsurf=use_repsurf

        surf_c = 10  # 用了 p 项就是 10，否则 9

        # --- LPR 维度（保持你原来的 13D） ---
        shape_c = 13
        in_c2d = in_channels + shape_c
        #RepSurf 分支
        if self.use_repsurf:
            self.surf_pool = AttnPoolK(surf_c, out_channels, dropout, norm)

        # --- 聚合器：AttnPool-K 替换 "共享MLP + MAX" ---
        if use_attnpool:
            self.agg = AttnPoolK(in_c2d, out_channels, dropout, norm=norm)
        else:
            self.mlp = nn.Sequential(
                nn.Conv2d(in_c2d, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels) if norm == "bn" else nn.Identity(),
                act_layer(inplace=True),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            )

        # --- 残差支路 ---
        if use_residual:
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm1d(out_channels) if norm == "bn" else nn.Identity(),
                )
            else:
                self.shortcut = nn.Identity()
            self.act = act_layer(inplace=True)

        # --- LFCM learnable mapping along K ---
        self.wK = nn.Linear(k, k, bias=False) if use_corr else None
        if self.wK is not None:
            nn.init.eye_(self.wK.weight)

        # --- LAN-IDKNN 的度量头：首层仅 xyz，后续层融合 feat ---
        if use_metric_knn:
            self.metric_head = AnisoMetric( 
                in_c=in_channels if in_channels != 3 else 3,
                use_feat=(in_channels != 3)
            )
        else:
            self.metric_head = None

    @torch.no_grad()
    def build_idx(self, base, xyz=None): 
        """
        若启用 use_metric_knn：
          - 首层： base应为xyz（或传None），仅用xyz建图；
          - 后续层： base为上一层特征，xyz需提供以融合几何项。
        若未启用：退回原 knn_with_dilation(base, k, dilation)。
        """
        k, d = self.k, self.dilation
        # 原始回退：不用可学习度量
        if not self.use_metric_knn or self.metric_head is None:
            idx, _ = knn_with_dilation(base, k, d)
            return idx

        # 可学习各向异性度量
        if (xyz is None) and (base is not None) and (base.shape[1] == 3):
            xyz = base
        assert xyz is not None, "LAN-IDKNN 需要提供 xyz 以计算几何距离。"

        s, beta = self.metric_head(base if self.metric_head.use_feat else xyz)  # s: (B,3,N) - 每个点在x,y,z方向的缩放因子。beta: 标量 - 几何距离和特征距离的融合权重
        geo = _metric_neg_sqdist_xyz(xyz, s)                                    # (B,N,N)
        pair_distance = geo
        if self.metric_head.use_feat and (base is not None):
            pair_distance = pair_distance + beta * _neg_sqdist_feat(base)

        w = k // d
        assert w > 1
        _, idxall = pair_distance.topk(k=k * d, dim=-1, largest=True)

        if d == 1:
            retidx = idxall[:, :, :k]
        elif d == 2:
            retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w + 2 * (k - w) + 1:2]), dim=-1)
        elif d == 3:
            retidx = torch.cat((idxall[:, :, :w],
                                 idxall[:, :, w + 1:w * 3:2],
                                 idxall[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]), dim=-1)
        elif d == 4:
            retidx = torch.cat((idxall[:, :, :w],
                                 idxall[:, :, w + 1:w * 3:2],
                                 idxall[:, :, 3 * w + 2: 6 * w:3],
                                 idxall[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
        else:
            retidx = idxall[:, :, ::d][:, :, :k]
        return retidx

    def forward(self, x, xyz, idx=None):
        """
        x:   (B,C,N)
        xyz: (B,3,N)
        idx: (B,N,K)  可选；若 None 且启用 LAN-IDKNN，将在本层内自动建图
        """
        B, C, N = x.shape
        if idx is None:
            # 自动建图（LAN-IDKNN 或原 KNN）
            base_for_metric = x if (self.metric_head is not None and self.metric_head.use_feat) else xyz
            idx = self.build_idx(base_for_metric, xyz=xyz)

        # 邻居特征/中心特征
        x_j = gather_feat_neighbors(x, idx)                 # (B,C,N,K)
        x_i = x.unsqueeze(-1).expand(-1, -1, -1, self.k)    # (B,C,N,K)
        feat_rel = x_j - x_i
        #RepSurf分支
        if self.use_repsurf:
            # 复用主分支的 idx 构造雨伞三角片（无需二次 KNN）
            xyz_bnc = xyz.transpose(1, 2)  # (B,N,3)
            umb = group_by_umbrella_from_idx(xyz_bnc, idx, drop_self=True)  # (B,N,K-1,3,3)

            # 法向 / 中心 / 极坐标 / p
            g_nor = cal_normal(umb, random_inv=self.training, is_group=True)    # (B,N,K-1,3)
            g_ctr = cal_center(umb)                                             # (B,N,K-1,3)
            g_pol = xyz2sphere(g_ctr)                                           # (B,N,K-1,3)
            g_pos = cal_const(g_nor, g_ctr)                                     # (B,N,K-1,1)

            # 数值稳定（可选，但推荐）
            g_nor = torch.where(torch.isfinite(g_nor), g_nor, torch.zeros_like(g_nor))
            g_ctr = torch.where(torch.isfinite(g_ctr), g_ctr, torch.zeros_like(g_ctr))
            g_pol = torch.where(torch.isfinite(g_pol), g_pol, torch.zeros_like(g_pol))
            g_pos = torch.where(torch.isfinite(g_pos), g_pos, torch.zeros_like(g_pos))

            # 拼接 & 聚合到 (B,Co,N)
            g_feat = torch.cat([g_ctr, g_pol, g_nor, g_pos], dim=-1)            # (B,N,K-1,10)
            g_feat = g_feat.permute(0, 3, 1, 2).contiguous()                    # (B,10,N,K-1)
            f_surf = self.surf_pool(g_feat)                                     # (B,Co,N)



        # LPR 13D
        xyz_bnc, knn_xyz = gather_xyz_neighbors(xyz, idx)   # (B,N,3), (B,N,K,3)
        if self.shape_repr == "cylinder":
            lshape = LocalFeatureRepresentaion_cylinder(xyz_bnc, knn_xyz, nsample=self.k)  # (B,N,K,13)
        else:
            lshape, _, _ = LocalFeatureRepresentaion_polar(xyz_bnc, knn_xyz, return_dis=True)  # (B,N,K,13)
        lshape = lshape.permute(0, 3, 1, 2).contiguous()     # (B,13,N,K)

        # LFCM（沿K的相关注意力）
        if self.use_corr:
            xi = F.normalize(x_i, dim=1)
            xj = F.normalize(x_j, dim=1)
            corr = (xi * xj).sum(1, keepdim=True)           # (B,1,N,K)
            a = corr.squeeze(1).reshape(B * N, self.k)      # (B*N,K)
            a = self.wK(a)
            a = torch.softmax(a, dim=-1).view(B, 1, N, self.k)
            feat_rel = feat_rel * a + feat_rel              # 与论文式(2)一致

        # 聚合（AttnPool-K | 共享MLP+MAX）
        base = torch.cat([feat_rel, lshape], dim=1)          # (B,C+13,N,K)
        if self.use_attnpool:
            f = self.agg(base)                               # (B,Co,N)
        else:
            f = self.mlp(base)                               # (B,Co,N,K)
            f = f.max(dim=-1, keepdim=False)[0]              # (B,Co,N)
        
        # 3) 与主干融合（f 是你主干已有输出）
        if self.use_repsurf:
            f = f + f_surf

        # 残差
        if self.use_residual:
            f = self.act(f + self.shortcut(x))
        return f, idx

# -------------------- 你现有的 LPR & dilation-KNN（原样复用） --------------------
def LocalFeatureRepresentaion_polar(xyz,knn_points,return_dis=True):
    b,n,k,_ = knn_points.shape
    eps=1e-9
    knn_points_norm = knn_points - xyz.unsqueeze(-2)
    local_dis = torch.sqrt(torch.sum(knn_points_norm **2 ,dim=-1)+eps)
    local_x = knn_points_norm[:,:,:,0]
    local_y = knn_points_norm[:,:,:,1]
    local_z = knn_points_norm[:,:,:,2]
    local_xy = torch.sqrt(local_x ** 2 + local_y ** 2+eps)
    local_xz = torch.sqrt(local_x ** 2 + local_z ** 2+eps)
    local_yz = torch.sqrt(local_y ** 2 + local_z ** 2+eps)
    z_fi = torch.atan2(local_y,local_x)
    z_ceta = torch.atan2(local_z,local_xy)
    y_fi = torch.atan2(local_z,local_x)
    y_ceta = torch.atan2(local_y,local_xz)
    x_fi = torch.atan2(local_z,local_y)
    x_ceta = torch.atan2(local_x,local_yz)
    xyzlift = xyz.unsqueeze(-2).repeat(1,1,k,1)
    if return_dis:
        local_feature = torch.cat((local_dis.unsqueeze(-1),z_fi.unsqueeze(-1),z_ceta.unsqueeze(-1),
                                   y_fi.unsqueeze(-1),y_ceta.unsqueeze(-1),x_fi.unsqueeze(-1),x_ceta.unsqueeze(-1),
                                   knn_points_norm,xyzlift),dim = -1)
    else:
        local_feature = torch.cat((z_fi.unsqueeze(-1),z_ceta.unsqueeze(-1),y_fi.unsqueeze(-1),
                                   y_ceta.unsqueeze(-1),x_fi.unsqueeze(-1),x_ceta.unsqueeze(-1),
                                   knn_points_norm,xyzlift), dim=-1)
    return local_feature,local_dis.unsqueeze(-1),knn_points_norm

def LocalFeatureRepresentaion_cylinder(xyz,knn_points,nsample,dila3 = False):
    knn_points_norm = knn_points - xyz.unsqueeze(-2)
    eps=1e-9
    local_dis = torch.sqrt(torch.sum(knn_points_norm **2 ,dim=-1)+eps)
    z_z = knn_points_norm[:,:,:,2]
    z_r = torch.sqrt(knn_points_norm[:,:,:,0] ** 2 + knn_points_norm[:,:,:,1] ** 2+eps)
    z_ceta = torch.atan2(knn_points_norm[:,:,:,1],knn_points_norm[:,:,:,0])
    y_y = knn_points_norm[:,:,:,1]
    y_r = torch.sqrt(knn_points_norm[:,:,:,0] ** 2 + knn_points_norm[:,:,:,2] ** 2+eps)
    y_ceta = torch.atan2(knn_points_norm[:,:,:,0],knn_points_norm[:,:,:,2])
    x_x = knn_points_norm[:,:,:,0]
    x_r = torch.sqrt(knn_points_norm[:,:,:,1] ** 2 + knn_points_norm[:,:,:,2] ** 2+eps)
    x_ceta = torch.atan2(knn_points_norm[:,:,:,2],knn_points_norm[:,:,:,1])
    xyz_lift = xyz.unsqueeze(-2).repeat(1,1,nsample if not dila3 else 3*nsample,1)
    local_features = torch.cat((z_z.unsqueeze(-1),y_y.unsqueeze(-1),x_x.unsqueeze(-1),
                                z_r.unsqueeze(-1),z_ceta.unsqueeze(-1),
                                y_r.unsqueeze(-1),y_ceta.unsqueeze(-1),
                                x_r.unsqueeze(-1),x_ceta.unsqueeze(-1),
                                local_dis.unsqueeze(-1),xyz_lift),dim = -1)
    return local_features

def knn_with_dilation(x, k, d):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # b,n,n
    xx = torch.sum(x ** 2, dim=1, keepdim=True)      # b,1,n
    pair_distance = -xx - inner - xx.transpose(2, 1) # b,n,n
    w = k // d
    assert w > 1
    feature_distance = pair_distance.topk(k=k * d, dim=-1)[0]
    idxall = pair_distance.topk(k=k * d, dim=-1)[1]
    if d == 1:
        retidx = idxall[:, :, :k]
    elif d == 2:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w + 2 * (k - w) + 1:2]), dim=-1)
    elif d == 3:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2],
                            idxall[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]), dim=-1)
    elif d == 4:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2],
                            idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
    elif d == 5:
        retidx = torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w * 3:2],
                            idxall[:, :, 3 * w + 2: 6 * w:3],
                            idxall[:, :, 6 * w + 3: w * 10:4],
                            idxall[:, :, 10 * w + 4:5 * (k - 4 * w) + 10 * w + 1:5]), dim=-1)
    return retidx, feature_distance

def group_by_umbrella(xyz, new_xyz, k=9, cuda=False):
    """
    Group a set of points into umbrella surfaces

    """
    idx = query_knn_point(k, xyz, new_xyz, cuda=cuda)
    torch.cuda.empty_cache()
    group_xyz = index_points(xyz, idx, cuda=cuda, is_group=True)[:, :, 1:]  # [B, N', K-1, 3]
    torch.cuda.empty_cache()

    group_xyz_norm = group_xyz - new_xyz.unsqueeze(-2)
    group_phi = xyz2sphere(group_xyz_norm)[..., 2]  # [B, N', K-1]
    sort_idx = group_phi.argsort(dim=-1)  # [B, N', K-1]

    # [B, N', K-1, 1, 3]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx).unsqueeze(-2)
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-3)
    group_centriod = torch.zeros_like(sorted_group_xyz)
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz, sorted_group_xyz_roll], dim=-2)

    return umbrella_group_xyz

def cal_normal(group_xyz, random_inv=False, is_group=False):
    """
    Calculate Normal Vector (Unit Form + First Term Positive)

    :param group_xyz: [B, N, K=3, 3] / [B, N, G, K=3, 3]
    :param random_inv:
    :param return_intersect:
    :param return_const:
    :return: [B, N, 3]
    """
    edge_vec1 = group_xyz[..., 1, :] - group_xyz[..., 0, :]  # [B, N, 3]
    edge_vec2 = group_xyz[..., 2, :] - group_xyz[..., 0, :]  # [B, N, 3]

    nor = torch.cross(edge_vec1, edge_vec2, dim=-1)
    den = torch.norm(nor, dim=-1, keepdim=True).clamp_min(1e-8)
    unit_nor = nor / den# [B, N, 3] / [B, N, G, 3]
    if not is_group:
        pos_mask = (unit_nor[..., 0] > 0).float() * 2. - 1.  # keep x_n positive
    else:
        pos_mask = (unit_nor[..., 0:1, 0] > 0).float() * 2. - 1.
    unit_nor = unit_nor * pos_mask.unsqueeze(-1)

    # batch-wise random inverse normal vector (prob: 0.5)
    if random_inv:
        random_mask = torch.randint(0, 2, (group_xyz.size(0), 1, 1)).float() * 2. - 1.
        random_mask = random_mask.to(unit_nor.device)
        if not is_group:
            unit_nor = unit_nor * random_mask
        else:
            unit_nor = unit_nor * random_mask.unsqueeze(-1)

    return unit_nor

def cal_center(group_xyz):
    """
    Calculate Global Coordinates of the Center of Triangle

    :param group_xyz: [B, N, K, 3] / [B, N, G, K, 3]; K >= 3
    :return: [B, N, 3] / [B, N, G, 3]
    """
    center = torch.mean(group_xyz, dim=-2)
    return center

def xyz2sphere(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate

    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    :param xyz: [B, N, 3] / [B, N, G, 3]
    :return: (rho, theta, phi) [B, N, 3] / [B, N, G, 3]
    """
    rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    rho = torch.clamp(rho, min=0)  # range: [0, inf]
    ratio = (xyz[..., 2, None] / rho.clamp_min(1e-8)).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(ratio)
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi]
    # check nan
    idx = rho == 0
    theta[idx] = 0

    if normalize:
        theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    out = torch.cat([rho, theta, phi], dim=-1)
    return out

def cal_const(normal, center, is_normalize=True):
    """
    Calculate Constant Term (Standard Version, with x_normal to be 1)

    math::
        const = x_nor * x_0 + y_nor * y_0 + z_nor * z_0

    :param is_normalize:
    :param normal: [B, N, 3] / [B, N, G, 3]
    :param center: [B, N, 3] / [B, N, G, 3]
    :return: [B, N, 1] / [B, N, G, 1]
    """
    const = torch.sum(normal * center, dim=-1, keepdim=True)
    factor = torch.sqrt(torch.Tensor([3])).to(normal.device)
    const = const / factor if is_normalize else const

    return const

def group_by_umbrella_from_idx(xyz_bnc, idx, drop_self=True):
    """
    使用主分支的 idx 构造雨伞三角片
    xyz_bnc: (B,N,3)
    idx:     (B,N,K)
    return:  umbrellas [B, N, K-1, 3(points), 3(coord)]
             points 顺序依次为 [center(0,0,0), p_i, p_{i+1}]
    """
    B, N, _ = xyz_bnc.shape
    K = idx.shape[-1]

    # 取出 K 个邻居坐标
    nbrs = index_points(xyz_bnc, idx)               # (B,N,K,3)

    # 去掉自环：你的 knn_with_dilation / topk 产生的 idx 通常首列是自身
    if drop_self:
        nbrs = nbrs[:, :, 1:, :]                    # (B,N,K-1,3)
    else:
        nbrs = nbrs[:, :, :K-1, :]

    # 相对中心坐标（中心点 -> 原点）
    rel = nbrs - xyz_bnc.unsqueeze(2)               # (B,N,K-1,3)

    # 以极角 phi 排序
    phi = xyz2sphere(rel)[..., 2]                   # (B,N,K-1) in [0,1]
    sort_idx = phi.argsort(dim=-1)                  # (B,N,K-1)
    # 按 K-1 维重排
    sorted_rel = torch.gather(
        rel, 2, sort_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
    )                                               # (B,N,K-1,3)

    # 组三角片：[center(0), p_i, p_{i+1}]
    tri_pi   = sorted_rel.unsqueeze(-2)             # (B,N,K-1,1,3)
    tri_pip1 = torch.roll(tri_pi, shifts=-1, dims=2)# (B,N,K-1,1,3)
    tri_ctr  = torch.zeros_like(tri_pi)             # (B,N,K-1,1,3)

    umbrellas = torch.cat([tri_ctr, tri_pi, tri_pip1], dim=-2)  # (B,N,K-1,3,3)
    return umbrellas





