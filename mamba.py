import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# 1) Consistent Traverse Serialization (CTS) — 蛇形 Code_func
# ---------------------------
class ConsistentTraverseSerializer:
    """
    一致遍历序列化器（CTS）
    - 将 [B,3,N] 点坐标归一化到 [0,1) 后，量化到 Ng×Ng×Ng 网格；
    - 使用 2D 蛇形编码 Code_func(n1,n2)，再沿第三维递归，得到 3D 蛇形序列；
    - 支持 6 种轴顺序（xyz/xzy/yxz/yzx/zxy/zyx），以减少单一扫描方向的偏置。
    """
    ORDERS = {
        "xyz": (0, 1, 2),
        "xzy": (0, 2, 1),
        "yxz": (1, 0, 2),
        "yzx": (1, 2, 0),
        "zxy": (2, 0, 1),
        "zyx": (2, 1, 0),
    }

    def __init__(self, grid_size: int = 32, eps: float = 1e-6):
        self.grid_size = int(grid_size)
        self.eps = eps

    @torch.no_grad()
    def _normalize_to_unit_cube(self, pos: torch.Tensor) -> torch.Tensor:
        """
        将坐标按每个 batch 的包围盒归一化到 [0,1)。
        Args: pos [B,3,N]
        """
        B, _, N = pos.shape
        pmin = pos.amin(dim=2, keepdim=True)         # [B,3,1]
        pmax = pos.amax(dim=2, keepdim=True)         # [B,3,1]
        scale = (pmax - pmin).clamp_min(self.eps)    # 防止退化
        p = (pos - pmin) / scale                     # [0,1]
        p = (p - self.eps).clamp(0.0, 1.0 - 2*self.eps)  # 稍微远离边界
        return p

    @torch.no_grad()
    def serialize_points(self, pos: torch.Tensor, order: str = "xyz") -> torch.Tensor:
        """
        生成蛇形 CTS 的排序下标。
        Args:
            pos:   [B,3,N]
            order: 'xyz' | 'xzy' | 'yxz' | 'yzx' | 'zxy' | 'zyx'
        Returns:
            sorted_idx: [B,N]  每个 batch 一条全排列索引
        """
        assert order in self.ORDERS, f"order must be in {list(self.ORDERS.keys())}"
        B, _, N = pos.shape
        Ng = self.grid_size

        # 归一化 → 量化到网格坐标 {0..Ng-1}
        p = self._normalize_to_unit_cube(pos)
        cg = (p * Ng).floor().clamp(0, Ng - 1 - 1e-6).long()  # [B,3,N]

        # 轴重排
        ax = self.ORDERS[order]
        g1, g2, g3 = cg[:, ax[0]], cg[:, ax[1]], cg[:, ax[2]]   # 各 [B,N]

        # 2D 蛇形编码：Code_func(g1,g2) ∈ [0, Ng*Ng-1]（向量化）
        even2 = (g2 % 2 == 0)
        code2 = torch.where(
            even2, g2 * Ng + g1,
            (g2 + 1) * Ng - g1 - 1
        )  # [B,N], 每个 (g1,g2) 的蛇形行优先码

        # 3D 递归：沿 g3 蛇形扫描 blocks of size Ng*Ng
        Ng2 = Ng * Ng
        even3 = (g3 % 2 == 0)
        code3 = torch.where(
            even3, g3 * Ng2 + code2,
            (g3 + 1) * Ng2 - code2 - 1
        )  # [B,N]

        # 将 code3 升序排序得到一致遍历序列
        sorted_idx = torch.argsort(code3, dim=-1)    # [B,N]
        return sorted_idx


# ---------------------------
# 2) 简化且更稳的 Mamba/SSM 原型块（教学版）
#    - 采用 depthwise conv 作为选择性预处理
#    - SSM 使用对角稳定参数 A (per-channel × d_state)，B 随 x 自适应
#    - 支持双向外包（外层处理）
# ---------------------------
class TinyMambaBlock(nn.Module):
    """
    教学版的 Mamba 风格块：
    y = out_proj( sel_scan( act( depthwise_conv( x_in ) ) ) ⊙ act(z) )
    - 重点修正：dt/B 广播维度与稳定性；不依赖第三方包。
    """
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 4, expand_factor: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand_factor * d_model)

        # 输入投影 -> (x, z) 两条分支
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        # depthwise 1D 卷积（选择性预处理）
        self.dw_conv = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=d_conv - 1, groups=self.d_inner)
        # 状态空间相关投影
        #   dt: [B,L,d_inner]  (正数)
        #   Bx: [B,L,d_inner*d_state]  (自适应输入到状态的增益)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        self.Bx_proj = nn.Linear(self.d_inner, self.d_inner * self.d_state)

        # 对角稳定参数 A（对角元 < 0），以及残差 D
        self.A_log = nn.Parameter(torch.randn(self.d_inner, self.d_state))   # A = -exp(A_log)
        self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state))       # 输出投影到通道
        self.D = nn.Parameter(torch.ones(self.d_inner))                      # 残差系数

        self.activation = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            y: [B, L, D]
        """
        B, L, D = x.shape
        xz = self.in_proj(x)                 # [B,L,2*I]
        x_in, z = xz.chunk(2, dim=-1)        # [B,L,I], [B,L,I]

        # depthwise conv 期望在序列维度上
        x_in = x_in.transpose(1, 2)          # [B,I,L]
        x_in = self.dw_conv(x_in)[:, :, :L]  # [B,I,L]（裁去 padding 溢出）
        x_in = x_in.transpose(1, 2)          # [B,L,I]
        x_in = self.activation(x_in)

        # 选择性扫描（向前）
        y_f = self._selective_scan(x_in)     # [B,L,I]
        # 门控
        y_f = y_f * self.activation(z)
        # 输出投影
        y = self.out_proj(y_f)               # [B,L,D]
        return y

    def _selective_scan(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        教学版逐时刻扫描（O(L) 循环）；在真实训练中建议替换为并行等价卷积实现。
        Args: x_in [B,L,I]
        Returns:    [B,L,I]
        """
        B, L, I = x_in.shape
        S = self.d_state

        # dt ≥ 0
        dt = F.softplus(self.dt_proj(x_in))                    # [B,L,I]
        # A < 0, 按通道×状态维度广播
        A = -torch.exp(self.A_log).to(x_in.dtype)              # [I,S]
        # Bx: 从输入到状态的增益
        Bx = self.Bx_proj(x_in).view(B, L, I, S)               # [B,L,I,S]
        # 输出权重 C
        C = self.C.to(x_in.dtype).unsqueeze(0).unsqueeze(0)    # [1,1,I,S]
        D = self.D.to(x_in.dtype).unsqueeze(0).unsqueeze(0)    # [1,1,I]

        # 扫描状态 h
        h = x_in.new_zeros(B, I, S)                            # [B,I,S]
        y_list = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1)                      # [B,I,1]
            x_t  = x_in[:, t].unsqueeze(-1)                    # [B,I,1]
            B_t  = Bx[:, t]                                    # [B,I,S]

            # 离散化：e^{A*dt}
            dA = torch.exp(A.unsqueeze(0) * dt_t)              # [B,I,S]
            # dB * x_t
            dBxt = (dt_t * B_t) * x_t                          # [B,I,S]

            # 状态更新
            h = h * dA + dBxt
            # 输出：sum(h * C, state) + D * x
            y_t = (h * C).sum(dim=-1) + (D.squeeze(-1) * x_in[:, t])
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)                         # [B,L,I]
        return y


# ---------------------------
# 3) 点云 Mamba 层（替代全局 Transformer）
#    - 单层仅用一个序列化顺序；建议在网络“跨层轮换”不同顺序
#    - 在序列首尾拼接 Order Prompts（可学习 tokens → Linear → 对齐通道）
#    - 3D 坐标线性位置编码（建议同一阶段共享这个 Linear）
#    - 双向：正向/反向各跑一次 TinyMambaBlock，结果取均值
# ---------------------------
class PointCloudMambaLayer(nn.Module):
    """
    使用 Mamba/SSM 进行全局建模的点云层（替代全局 Transformer 块）
    Args:
        channels:         通道数 C
        order:            本层使用的序列化顺序（'xyz' 等），建议不同层轮换
        num_prompts:      每侧提示 token 数 Np（首部 Np + 尾部 Np）
        prompt_dim:       提示 token 原始维度（会通过 Linear 投到 C）
        grid_size:        CTS 网格边长 Ng
        d_state, d_conv, expand_factor: TinyMambaBlock 的超参
        pos_encoding:     可外部传入（以便“阶段内共享”），否则内部创建
        prompt_proj:      可外部传入（阶段内共享），否则内部创建
    """
    def __init__(
        self,
        channels: int,
        order: str = "xyz",
        num_prompts: int = 6,
        prompt_dim: int = 64,
        grid_size: int = 32,
        d_state: int = 8,
        d_conv: int = 4,
        expand_factor: float = 2.0,
        pos_encoding: nn.Module = None,
        prompt_proj: nn.Module = None,
    ):
        super().__init__()
        self.C = channels
        self.order = order
        self.num_prompts = num_prompts

        # CTS
        self.serializer = ConsistentTraverseSerializer(grid_size=grid_size)

        # 位置编码（建议同一阶段共享）
        self.pos_encoding = pos_encoding or nn.Linear(3, channels)

        # Order Prompts：为 6 种顺序各准备 Np 个提示（原始维 prompt_dim），
        # 再用线性层映射到本层通道维（建议阶段内共享 prompt_proj）
        self.orders = list(ConsistentTraverseSerializer.ORDERS.keys())
        self.order_to_idx = {o: i for i, o in enumerate(self.orders)}
        self.prompt_table = nn.Embedding(len(self.orders) * num_prompts, prompt_dim)
        self.prompt_proj = prompt_proj or nn.Linear(prompt_dim, channels, bias=False)

        # 单层单序列的 TinyMamba（双向外包）
        self.mamba = TinyMambaBlock(d_model=channels, d_state=d_state,
                                    d_conv=d_conv, expand_factor=expand_factor)

        # 归一化与残差
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU(inplace=True)

    def _make_prompts(self, B: int, device: torch.device) -> torch.Tensor:
        """生成并映射到通道维的首尾提示 tokens；返回 [B, 2*Np, C]"""
        order_idx = self.order_to_idx[self.order]
        base = order_idx * self.num_prompts
        idx = torch.arange(base, base + self.num_prompts, device=device, dtype=torch.long)  # [Np]
        tokens = self.prompt_table(idx)                 # [Np, P]
        tokens = self.prompt_proj(tokens)               # [Np, C]
        tokens = tokens.unsqueeze(0).expand(B, -1, -1)  # [B,Np,C]
        # 首尾各一份
        return torch.cat([tokens, tokens], dim=1)       # [B,2*Np,C]

    @staticmethod
    def _gather_by_index(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """feat: [B,C,N], idx: [B,N] -> out: [B,C,N]（按 idx 重排）"""
        B, C, N = feat.shape
        idx_exp = idx.unsqueeze(1).expand(-1, C, -1)    # [B,C,N]
        return torch.gather(feat, dim=2, index=idx_exp)

    @staticmethod
    def _invert_permutation(idx: torch.Tensor) -> torch.Tensor:
        """对每个 batch 求 idx 的逆置换；idx: [B,N] -> inv_idx: [B,N]"""
        B, N = idx.shape
        inv = torch.empty_like(idx)
        arangeN = torch.arange(N, device=idx.device).unsqueeze(0).expand(B, -1)  # [B,N]
        inv.scatter_(1, idx, arangeN)
        return inv

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:   [B,C,N] 点特征
            pos: [B,3,N] 点坐标（建议提供；否则会随机生成，训练不稳）
        Returns:
            out: [B,C,N]
        """
        B, C, N = x.shape
        device = x.device
        if pos is None:
            # 强烈建议传入真实坐标；这里仅兜底，避免崩溃
            pos = torch.randn(B, 3, N, device=device) * 0.5

        # 1) CTS：得到排序索引
        sorted_idx = self.serializer.serialize_points(pos, order=self.order)  # [B,N]
        inv_idx = self._invert_permutation(sorted_idx)                        # [B,N]

        # 2) 重排为序列形状 [B,N,C]，并加坐标线性位置编码
        x_sorted = self._gather_by_index(x, sorted_idx).transpose(1, 2)       # [B,N,C]
        pos_sorted = self._gather_by_index(pos, sorted_idx).transpose(1, 2)   # [B,N,3]
        x_seq = x_sorted + self.pos_encoding(pos_sorted)                       # [B,N,C]

        # 3) Order Prompts：序列首尾拼接 Np 个提示
        prompts = self._make_prompts(B, device)                                # [B,2*Np,C]
        Np2 = prompts.shape[1] // 2
        x_with_p = torch.cat([prompts[:, :Np2], x_seq, prompts[:, Np2:]], dim=1)  # [B,N+2Np,C]

        # 4) 双向 Mamba
        y_f = self.mamba(x_with_p)                                            # [B,N+2Np,C]
        y_b = self.mamba(torch.flip(x_with_p, dims=[1]))
        y_b = torch.flip(y_b, dims=[1])
        y_seq = 0.5 * (y_f + y_b)

        # 5) 去掉提示，恢复到原始点顺序
        y_seq = y_seq[:, Np2:-Np2, :]                                         # [B,N,C]
        y_sorted = y_seq.transpose(1, 2)                                      # [B,C,N]
        y = self._gather_by_index(y_sorted, inv_idx)                          # [B,C,N]

        # 6) 残差 + Norm
        out = x + y
        out = self.act(self.norm(out))
        return out


# ---------------------------
# 4) 用法示例（集成到你的网络中）
# ---------------------------
if __name__ == "__main__":
    B, C, N = 2, 256, 2048
    x = torch.randn(B, C, N).cuda() if torch.cuda.is_available() else torch.randn(B, C, N)
    pos = torch.randn(B, 3, N).cuda() if torch.cuda.is_available() else torch.randn(B, 3, N)

    # 建议：在四个全局层里轮换顺序，例如：
    # layer1: order='xyz'; layer2: 'xzy'; layer3: 'yxz'; layer4: 'yzx'
    layer = PointCloudMambaLayer(
        channels=C,
        order="xyz",
        num_prompts=6,
        prompt_dim=64,
        grid_size=32,
        d_state=8,
        d_conv=4,
        expand_factor=2.0,
    )
    layer = layer.cuda() if torch.cuda.is_available() else layer
    y = layer(x, pos)   # [B,C,N]
    print(y.shape)