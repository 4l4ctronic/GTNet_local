import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 你已有的：各向异性度量头 / 距离 ----------
class AnisoMetric(nn.Module):
    """为每个中心点预测 diag(sx,sy,sz)；beta 在 CAN 首层不用特征时可以忽略"""
    def __init__(self, in_c=3, use_feat=False):
        super().__init__()
        self.use_feat = use_feat
        self.enc = nn.Sequential(
            nn.Conv1d(in_c, 64, 1), nn.ReLU(True),
            nn.Conv1d(64, 32, 1),  nn.ReLU(True),
            nn.Conv1d(32, 3, 1)    # -> (B,3,N)
        )
        self.beta = nn.Parameter(torch.tensor(0.5))  # 仅后续层用到

    def forward(self, x_or_xyz):             # (B,C_or_3,N)
        s = F.softplus(self.enc(x_or_xyz)) + 1e-3    # (B,3,N) 逐点正定缩放
        beta = F.softplus(self.beta)
        return s, beta

def _metric_neg_sqdist_xyz(xyz, s):          # (B,3,N),(B,3,N)->(B,N,N)
    term1 = (s * (xyz**2)).sum(1, keepdim=True)        # (B,1,N)
    term2 = -2 * ((s * xyz).transpose(2,1) @ xyz)      # (B,N,N)
    term3 = s.transpose(2,1) @ (xyz**2)                # (B,N,N)
    D = term1.transpose(1,2) + term2 + term3           # (B,N,N)
    return -D                                          # 越大越近

# ---------- 辅助：按 dilation 选 k 个邻居 ----------
def _select_dilated(idxall, k, d):
    # idxall: (B,N,k*d) from topk (largest=True)
    if d == 1:
        return idxall[:, :, :k]
    w = k // d
    assert w > 1, "k 必须 >= d*2 才有意义"
    if d == 2:
        return torch.cat((idxall[:, :, :w], idxall[:, :, w + 1:w + 2 * (k - w) + 1:2]), dim=-1)
    if d == 3:
        return torch.cat((idxall[:, :, :w],
                          idxall[:, :, w + 1:w * 3:2],
                          idxall[:, :, w * 3 + 2:(k - 2 * w) * 3 + 3 * w + 1:3]), dim=-1)
    if d == 4:
        return torch.cat((idxall[:, :, :w],
                          idxall[:, :, w + 1:w * 3:2],
                          idxall[:, :, 3 * w + 2: 6 * w:3],
                          idxall[:, :, 6 * w + 3:(k - 3 * w) * 4 + 6 * w + 1:4]), dim=-1)
    # Fallback：步长采样
    return idxall[:, :, ::d][:, :, :k]

# ---------- 改造后的 CAN：内部用 LAN-IDKNN 建图 ----------
class CAN_LANIDKNN(nn.Module):
    """
    输入:  xyz (B,3,N)
    输出:  T   (B,3,3)  —— 用于坐标对齐的仿射/近似旋转矩阵
    说明:  用 AnisoMetric 预测逐点尺度 s，基于 s 的各向异性几何距离做 top-k（含 dilation）。
          构造 6 通道边特征 [x_j - x_i, x_i] 后，结构与原 Transform_Net 一致。
    """
    def __init__(self, k=20, dilation=1):
        super().__init__()
        self.k = k
        self.d = dilation

        # 度量头：CAN 首层只用几何，不融合特征
        self.metric_head = AnisoMetric(in_c=3, use_feat=False)

        # === 卷积主干（与原 Transform_Net 对齐；注意命名不重复） ===
        self.bn2d1 = nn.BatchNorm2d(64)
        self.bn2d2 = nn.BatchNorm2d(128)
        self.bn1d3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6,   64, 1, bias=False), self.bn2d1, nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), self.bn2d2, nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv1d(128,1024, 1, bias=False), self.bn1d3, nn.LeakyReLU(0.2, inplace=True))

        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.to_T = nn.Linear(256, 3 * 3)
        nn.init.constant_(self.to_T.weight, 0.0)
        nn.init.eye_(self.to_T.bias.view(3, 3))

    @torch.no_grad()
    def _build_idx(self, xyz):
        """LAN-IDKNN：仅用几何算对称可分解的各向异性距离，然后 dilation 取邻。"""
        # xyz: (B,3,N)
        s, _ = self.metric_head(xyz)                 # (B,3,N)
        pair = _metric_neg_sqdist_xyz(xyz, s)        # (B,N,N)  大=近
        idxall = pair.topk(k=self.k * self.d, dim=-1, largest=True)[1]  # (B,N,k*d)
        return _select_dilated(idxall, self.k, self.d)                  # (B,N,k)

    def _edge6(self, xyz, idx):
        """根据 idx 取邻，构造 [x_j-x_i, x_i] 边特征为 (B,6,N,K)"""
        # xyz: (B,3,N); idx: (B,N,K)
        B, _, N = xyz.shape
        K = idx.shape[-1]
        device = xyz.device

        xyz_bnc = xyz.transpose(1, 2).contiguous()               # (B,N,3)
        idx_base = torch.arange(B, device=device).view(-1,1,1) * N
        nbr = xyz_bnc.reshape(B * N, 3)[(idx + idx_base).view(-1), :].view(B, N, K, 3)  # (B,N,K,3)
        ctr = xyz_bnc.view(B, N, 1, 3).expand(-1, -1, K, -1)                               # (B,N,K,3)

        feat = torch.cat((nbr - ctr, ctr), dim=3).permute(0, 3, 1, 2).contiguous()        # (B,6,N,K)
        return feat

    def forward(self, xyz):
        """
        xyz: (B,3,N)
        return:
            T: (B,3,3)
        """
        idx = self._build_idx(xyz)            # (B,N,K)
        F6  = self._edge6(xyz, idx)           # (B,6,N,K)

        x = self.conv1(F6)                    # (B,64,N,K)
        x = self.conv2(x)                     # (B,128,N,K)
        x = x.max(dim=-1)[0]                  # (B,128,N)

        x = self.conv3(x)                     # (B,1024,N)
        x = x.max(dim=-1)[0]                  # (B,1024)

        x = F.leaky_relu(self.bn_fc1(self.fc1(x)), 0.2)   # (B,512)
        x = F.leaky_relu(self.bn_fc2(self.fc2(x)), 0.2)   # (B,256)

        T = self.to_T(x).view(-1, 3, 3)       # (B,3,3)
        return T

# ---------- 可选：正交正则，放进你的 loss 里 ----------
def ortho_regularizer(T):
    # T: (B,3,3)
    I = torch.eye(3, device=T.device).unsqueeze(0).expand(T.size(0), -1, -1)
    return ((I - T @ T.transpose(2,1))**2).sum(dim=(1,2)).mean()
