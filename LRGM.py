import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_c, out_c, dropout=0.0, norm="bn"):
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
                 use_attnpool=True, use_metric_knn=True):
        super().__init__()
        self.k = k
        self.dilation = dilation
        self.shape_repr = shape_repr
        self.use_corr = use_corr
        self.use_residual = use_residual
        self.use_attnpool = use_attnpool
        self.use_metric_knn = use_metric_knn

        # --- LPR 维度（保持你原来的 13D） ---
        shape_c = 13
        in_c2d = in_channels + shape_c

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
