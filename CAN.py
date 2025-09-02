class CAN(nn.Module):
    """Coordinate Adjustment Network: 回归 tdim×tdim 变换矩阵（默认 3×3）"""
    def __init__(self, tdim: int = 3):
        super().__init__()
        self.tdim = tdim
        # 输入是边特征 (B, 2*tdim, N, K) —— 例如 tdim=3 时为 6 通道
        self.conv1 = nn.Sequential(nn.Conv2d(2*tdim, 64, 1, bias=False),
                                   nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False),
                                   nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1, bias=False),
                                   nn.BatchNorm1d(1024), nn.LeakyReLU(0.2, True))
        self.fc1 = nn.Sequential(nn.Linear(1024, 512, bias=False),
                                 nn.BatchNorm1d(512), nn.LeakyReLU(0.2, True))
        self.fc2 = nn.Sequential(nn.Linear(512, 256, bias=False),
                                 nn.BatchNorm1d(256), nn.LeakyReLU(0.2, True))
        self.transform = nn.Linear(256, tdim * tdim)
        nn.init.zeros_(self.transform.weight)
        with torch.no_grad():
            self.transform.bias.copy_(torch.eye(tdim).reshape(-1))

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        # edge_feat: (B, 2*tdim, N, K)
        x = self.conv1(edge_feat)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=False)[0]  # (B,128,N)
        x = self.conv3(x)                    # (B,1024,N)
        x = x.max(dim=-1, keepdim=False)[0]  # (B,1024)
        x = self.fc1(x)                      # (B,512)
        x = self.fc2(x)                      # (B,256)
        T = self.transform(x).view(-1, self.tdim, self.tdim)  # (B, tdim, tdim)
        return T


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: torch.Tensor = None):
    """
    x: (B, C, N). 返回:
      feat: (B, 2C, N, K)  —— [xj - xi, xi] 的 EdgeConv 特征
      idx:  (B, N, K)      —— 批内 KNN 索引
    """
    B, C, N = x.shape
    device = x.device

    if idx is None:
        # 这里假设 knn(x, k) 已实现，基于当前特征做 KNN
        idx = knn(x, k=k)  # (B, N, K)

    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N  # (B,1,1)
    idx_flat = (idx + idx_base).reshape(-1)  # (B*N*K,)

    x_t = x.transpose(2, 1).contiguous()          # (B, N, C)
    neighbors = x_t.reshape(B * N, C)[idx_flat, :]\
                  .view(B, N, k, C)               # (B, N, K, C)
    central = x_t.view(B, N, 1, C).expand(-1, -1, k, -1)

    feat = torch.cat([neighbors - central, central], dim=3)\
             .permute(0, 3, 1, 2).contiguous()    # (B, 2C, N, K)

    return feat, idx
