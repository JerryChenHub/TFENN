import torch
import math
import torch.nn as nn

from core.layers import (
    D6EquivxxLinear,
    D6EquivVectorBlockGating,
    D6EquivRRLinear,
    D6EquivRRBlockGating,
    D6EquivRR_GroupConv,
    D6EquivRR_SVDActivation,
    D6EquivxxGroupConvDense,
    D6LeftEquivRightInv_TFENNGroupConv,
    D6LeftEquivRightInv_GroupConv, D6EquivRRBlockGatingTH,
    D6EquivRRBlockGatingV2, D6EquivRR_SpectralPolyGate
)


def _init_linear_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class StandardMLP(nn.Module):
    """
    A plain MLP that maps flat features to targets.
    Default: in_dim=12 (R:9 + x:3), out_dim=6 (F:3 + M:3)
    """
    def __init__(
        self,
        in_dim: int = 12,
        out_dim: int = 6,
        hidden_dim: int = 128,
        num_hidden_layers: int = 4,
        activation_fn: nn.Module=nn.LeakyReLU()
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation_fn)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(_init_linear_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        return self.net(x)


class TFENN_1(nn.Module):
    """
    x->x->x & R->R->R->x  Then group conv on x
    """
    def __init__(self,
                 x_in_channels: int = 1,
                 R_in_channels: int = 1,
                 x_hidden_channels: int = 32,
                 R_hidden_channels: int = 16,
                 num_x_layers: int = 2,
                 num_R_layers: int = 2,
                 R2x_channels: int = 32,
                 out_channels: int = 2,
                 activation=nn.Sigmoid(),
                 head_activation=nn.ELU()):
        super().__init__()
        assert num_x_layers >= 0 and num_R_layers >= 0

        # ---- x->x->x ----
        self.x_branch = nn.ModuleList()
        cx = x_in_channels
        for _ in range(num_x_layers):
            self.x_branch.append(D6EquivxxLinear(cx, x_hidden_channels))
            self.x_branch.append(D6EquivVectorBlockGating(act=activation))
            cx = x_hidden_channels

        # ---- R->R->R ----
        self.R_branch = nn.ModuleList()
        cR = R_in_channels
        for _ in range(num_R_layers):
            self.R_branch.append(D6EquivRRLinear(cR, R_hidden_channels))
            self.R_branch.append(D6EquivRRBlockGating(act=activation))
            cR = R_hidden_channels

        self.R2x = D6LeftEquivRightInv_TFENNGroupConv(cR, R2x_channels,act=activation)

        fused_in = cx + R2x_channels
        self.head = D6EquivxxGroupConvDense(fused_in, out_channels,activation=head_activation)

    @staticmethod
    def _normalize_shapes(x: torch.Tensor, R: torch.Tensor):
        if x.dim() == 2 and x.size(-1) == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(-1) == 3, "x must be (B,Cx,3) or (B,3)"
        if R.dim() == 3 and R.size(-2) == 3 and R.size(-1) == 3:
            R = R.unsqueeze(1)
        assert R.dim() == 4 and R.size(-2) == 3 and R.size(-1) == 3, \
            "R must be (B,Cr,3,3) or (B,3,3)"
        return x, R

    def forward(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        x, R = self._normalize_shapes(x, R)

        # x-branch
        for layer in self.x_branch:
            x = layer(x)

        # R-branch
        for layer in self.R_branch:
            R = layer(R)

        # R -> x
        R_vec = self.R2x(R)

        x_fused = torch.cat([x, R_vec], dim=1)
        out = self.head(x_fused)  # (B, out_channels, 3)
        return out


class TFENN_2(nn.Module):
    """
    x->x->x & R->R->R->x  Then group conv on x
    """
    def __init__(self,
                 x_in_channels: int = 1,
                 R_in_channels: int = 1,
                 x_hidden_channels: int = 32,
                 R_hidden_channels: int = 16,
                 num_x_layers: int = 2,
                 num_R_layers: int = 2,
                 R2x_channels: int = 32,
                 out_channels: int = 2,
                 activation=nn.Sigmoid(),
                 head_activation=nn.ELU(),
                 num_xx_head_layers: int = 2,
                 xx_head_hidden_channels: int | None = None
                 ):
        super().__init__()
        assert num_x_layers >= 0 and num_R_layers >= 0

        # ---- x->x->x ----
        self.x_branch = nn.ModuleList()
        cx = x_in_channels
        for _ in range(num_x_layers):
            self.x_branch.append(D6EquivxxLinear(cx, x_hidden_channels))
            self.x_branch.append(D6EquivVectorBlockGating(act=activation))
            cx = x_hidden_channels

        # ---- R->R->R ----
        self.R_branch = nn.ModuleList()
        cR = R_in_channels
        for _ in range(num_R_layers):
            self.R_branch.append(D6EquivRRLinear(cR, R_hidden_channels))
            self.R_branch.append(D6EquivRRBlockGating(act=activation))
            cR = R_hidden_channels

        self.R2x = D6LeftEquivRightInv_TFENNGroupConv(cR, R2x_channels,act=activation)

        fused_in = cx + R2x_channels
        if xx_head_hidden_channels is None:
            xx_head_hidden_channels = fused_in

        head_layers = []
        in_c = fused_in
        for _ in range(max(0, num_xx_head_layers - 1)):
            head_layers.append(D6EquivxxGroupConvDense(in_c, xx_head_hidden_channels,
                                                       activation=head_activation))
            in_c = xx_head_hidden_channels
        head_layers.append(D6EquivxxGroupConvDense(in_c, out_channels,
                                                   activation=head_activation))
        self.xx_head = nn.Sequential(*head_layers)

    @staticmethod
    def _normalize_shapes(x: torch.Tensor, R: torch.Tensor):
        if x.dim() == 2 and x.size(-1) == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(-1) == 3, "x must be (B,Cx,3) or (B,3)"
        if R.dim() == 3 and R.size(-2) == 3 and R.size(-1) == 3:
            R = R.unsqueeze(1)
        assert R.dim() == 4 and R.size(-2) == 3 and R.size(-1) == 3, \
            "R must be (B,Cr,3,3) or (B,3,3)"
        return x, R

    def forward(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        x, R = self._normalize_shapes(x, R)

        # x-branch
        for layer in self.x_branch:
            x = layer(x)

        # R-branch
        for layer in self.R_branch:
            R = layer(R)

        # R -> x
        R_vec = self.R2x(R)

        x_fused = torch.cat([x, R_vec], dim=1)
        out = self.xx_head(x_fused)  # (B, out_channels, 3)
        return out

class TFENN_2_SVD(nn.Module):
    """
    x->x->x & R->R->R->x  Then group conv on x
    """
    def __init__(self,
                 x_in_channels: int = 1,
                 R_in_channels: int = 1,
                 x_hidden_channels: int = 32,
                 R_hidden_channels: int = 16,
                 num_x_layers: int = 2,
                 num_R_layers: int = 2,
                 R2x_channels: int = 32,
                 out_channels: int = 2,
                 activation=nn.Sigmoid(),
                 head_activation=nn.ELU(),
                 num_xx_head_layers: int = 2,
                 xx_head_hidden_channels: int | None = None
                 ):
        super().__init__()
        assert num_x_layers >= 0 and num_R_layers >= 0

        # ---- x->x->x ----
        self.x_branch = nn.ModuleList()
        cx = x_in_channels
        for _ in range(num_x_layers):
            self.x_branch.append(D6EquivxxLinear(cx, x_hidden_channels))
            self.x_branch.append(D6EquivVectorBlockGating(act=activation))
            cx = x_hidden_channels

        # ---- R->R->R ----
        self.R_branch = nn.ModuleList()
        cR = R_in_channels
        for _ in range(num_R_layers):
            self.R_branch.append(D6EquivRRLinear(cR, R_hidden_channels))
            self.R_branch.append(D6EquivRR_SVDActivation(act=activation))
            cR = R_hidden_channels

        self.R2x = D6LeftEquivRightInv_TFENNGroupConv(cR, R2x_channels,act=activation)

        fused_in = cx + R2x_channels
        if xx_head_hidden_channels is None:
            xx_head_hidden_channels = fused_in

        head_layers = []
        in_c = fused_in
        for _ in range(max(0, num_xx_head_layers - 1)):
            head_layers.append(D6EquivxxGroupConvDense(in_c, xx_head_hidden_channels,
                                                       activation=head_activation))
            in_c = xx_head_hidden_channels
        head_layers.append(D6EquivxxGroupConvDense(in_c, out_channels,
                                                   activation=head_activation))
        self.xx_head = nn.Sequential(*head_layers)

    @staticmethod
    def _normalize_shapes(x: torch.Tensor, R: torch.Tensor):
        if x.dim() == 2 and x.size(-1) == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(-1) == 3, "x must be (B,Cx,3) or (B,3)"
        if R.dim() == 3 and R.size(-2) == 3 and R.size(-1) == 3:
            R = R.unsqueeze(1)
        assert R.dim() == 4 and R.size(-2) == 3 and R.size(-1) == 3, \
            "R must be (B,Cr,3,3) or (B,3,3)"
        return x, R

    def forward(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        x, R = self._normalize_shapes(x, R)

        # x-branch
        for layer in self.x_branch:
            x = layer(x)

        # R-branch
        for layer in self.R_branch:
            R = layer(R)

        # R -> x
        R_vec = self.R2x(R)

        x_fused = torch.cat([x, R_vec], dim=1)
        out = self.xx_head(x_fused)  # (B, out_channels, 3)
        return out

class TFENN_3(nn.Module):
    """
    x->x (0..L)  &  R->R (0..L)  &  R->x(GroupConv)  →  concat  →  x-head(GroupConv)
      x: (B, Cx, 3)
      R: (B, CR, 3, 3)
      y: (B, Cout, 3)
    """
    def __init__(self,
                 x_in_channels: int = 1,
                 R_in_channels: int = 1,
                 x_hidden_channels: int = 32,
                 R_hidden_channels: int = 16,
                 num_x_layers: int = 0,
                 num_R_layers: int = 0,
                 R2x_channels: int = 32,
                 out_channels: int = 2,
                 activation=nn.LeakyReLU(),
                 head_activation=nn.LeakyReLU(),
                 num_xx_head_layers: int = 2,
                 xx_head_hidden_channels: int | None = None,
                 xx_channels: int | None = None):
        super().__init__()
        assert num_x_layers >= 0 and num_R_layers >= 0

        # ---- x branch: (x->x)*num_x_layers ----
        self.x_branch = nn.ModuleList()
        cx = x_in_channels
        for _ in range(num_x_layers):
            out_c = xx_channels if (xx_channels is not None and _ == num_x_layers - 1) else x_hidden_channels
            self.x_branch.append(D6EquivxxLinear(cx, out_c))
            self.x_branch.append(D6EquivVectorBlockGating(act=activation))
            cx = out_c

        # ---- R branch: (R->R)*num_R_layers ----
        self.R_branch = nn.ModuleList()
        cR = R_in_channels
        for _ in range(num_R_layers):
            self.R_branch.append(D6EquivRRLinear(cR, R_hidden_channels))
            self.R_branch.append(D6EquivRRBlockGating(act=activation))
            cR = R_hidden_channels

        # ---- R->x: 纯 group convolution 版本 ----
        self.R2x = D6LeftEquivRightInv_GroupConv(cR, R2x_channels, act=activation)

        # ---- x-head (x->x): group conv 堆叠 ----
        fused_in = cx + R2x_channels
        if xx_head_hidden_channels is None:
            xx_head_hidden_channels = fused_in

        head_layers = []
        in_c = fused_in
        for _ in range(max(0, num_xx_head_layers - 1)):
            head_layers.append(D6EquivxxGroupConvDense(in_c, xx_head_hidden_channels,
                                                       activation=head_activation))
            in_c = xx_head_hidden_channels
        head_layers.append(D6EquivxxGroupConvDense(in_c, out_channels,
                                                   activation=head_activation))
        self.head = nn.Sequential(*head_layers)

        self.apply(_init_linear_weights)

    @staticmethod
    def _normalize_shapes(x: torch.Tensor, R: torch.Tensor):
        if x.dim() == 2 and x.size(-1) == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(-1) == 3, "x must be (B,Cx,3) or (B,3)"
        if R.dim() == 3 and R.size(-2) == 3 and R.size(-1) == 3:
            R = R.unsqueeze(1)
        assert R.dim() == 4 and R.size(-2) == 3 and R.size(-1) == 3, \
            "R must be (B,Cr,3,3) or (B,3,3)"
        return x, R

    def forward(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # x: (B, Cx, 3);  R: (B, CR, 3, 3)
        # x branch
        x, R = self._normalize_shapes(x, R)
        x_feat = x
        for layer in self.x_branch:
            x_feat = layer(x_feat)

        # R branch
        R_feat = R
        for layer in self.R_branch:
            R_feat = layer(R_feat)

        # R->x
        rx_feat = self.R2x(R_feat)  # (B, R2x_channels, 3)

        # fuse & head
        fused = torch.cat([x_feat, rx_feat], dim=1)  # (B, fused_in, 3)
        y = self.head(fused)                         # (B, out_channels, 3)
        return y


class TFENN_3_1(nn.Module):
    """
    x->x (0..L)  &  R->R (0..L)  &  R->x(GroupConv)  →  concat  →  x-head(GroupConv)
      x: (B, Cx, 3)
      R: (B, CR, 3, 3)
      y: (B, Cout, 3)
    """
    def __init__(self,
                 x_in_channels: int = 1,
                 R_in_channels: int = 1,
                 x_hidden_channels: int = 32,
                 R_hidden_channels: int = 16,
                 num_x_layers: int = 0,
                 num_R_layers: int = 0,
                 R2x_channels: int = 32,
                 out_channels: int = 2,
                 activation=nn.LeakyReLU(),
                 head_activation=nn.LeakyReLU(),
                 num_xx_head_layers: int = 2,
                 xx_head_hidden_channels: int | None = None,
                 xx_channels: int | None = None):
        super().__init__()
        assert num_x_layers >= 0 and num_R_layers >= 0

        # ---- x branch: (x->x)*num_x_layers ----
        self.x_branch = nn.ModuleList()
        cx = x_in_channels
        for _ in range(num_x_layers):
            out_c = xx_channels if (xx_channels is not None and _ == num_x_layers - 1) else x_hidden_channels
            self.x_branch.append(D6EquivxxLinear(cx, out_c))
            self.x_branch.append(D6EquivVectorBlockGating(act=activation))
            cx = out_c

        # ---- R branch: (R->R)*num_R_layers ----
        self.R_branch = nn.ModuleList()
        cR = R_in_channels
        for _ in range(num_R_layers):
            self.R_branch.append(D6EquivRRLinear(cR, R_hidden_channels))
            self.R_branch.append(D6EquivRRBlockGatingTH())
            cR = R_hidden_channels

        # ---- R->x: 纯 group convolution 版本 ----
        self.R2x = D6LeftEquivRightInv_GroupConv(cR, R2x_channels, act=activation)

        # ---- x-head (x->x): group conv 堆叠 ----
        fused_in = cx + R2x_channels
        if xx_head_hidden_channels is None:
            xx_head_hidden_channels = fused_in

        head_layers = []
        in_c = fused_in
        for _ in range(max(0, num_xx_head_layers - 1)):
            head_layers.append(D6EquivxxGroupConvDense(in_c, xx_head_hidden_channels,
                                                       activation=head_activation))
            in_c = xx_head_hidden_channels
        head_layers.append(D6EquivxxGroupConvDense(in_c, out_channels,
                                                   activation=head_activation))
        self.head = nn.Sequential(*head_layers)

        self.apply(_init_linear_weights)

    @staticmethod
    def _normalize_shapes(x: torch.Tensor, R: torch.Tensor):
        if x.dim() == 2 and x.size(-1) == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(-1) == 3, "x must be (B,Cx,3) or (B,3)"
        if R.dim() == 3 and R.size(-2) == 3 and R.size(-1) == 3:
            R = R.unsqueeze(1)
        assert R.dim() == 4 and R.size(-2) == 3 and R.size(-1) == 3, \
            "R must be (B,Cr,3,3) or (B,3,3)"
        return x, R

    def forward(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # x: (B, Cx, 3);  R: (B, CR, 3, 3)
        # x branch
        x, R = self._normalize_shapes(x, R)
        x_feat = x
        for layer in self.x_branch:
            x_feat = layer(x_feat)

        # R branch
        R_feat = R
        for layer in self.R_branch:
            R_feat = layer(R_feat)

        # R->x
        rx_feat = self.R2x(R_feat)  # (B, R2x_channels, 3)

        # fuse & head
        fused = torch.cat([x_feat, rx_feat], dim=1)  # (B, fused_in, 3)
        y = self.head(fused)                         # (B, out_channels, 3)
        return y

class TFENN_3_2(nn.Module):
    """
    x->x (0..L)  &  R->R (0..L)  &  R->x(GroupConv)  →  concat  →  x-head(GroupConv)
      x: (B, Cx, 3)
      R: (B, CR, 3, 3)
      y: (B, Cout, 3)
    """
    def __init__(self,
                 x_in_channels: int = 1,
                 R_in_channels: int = 1,
                 x_hidden_channels: int = 32,
                 R_hidden_channels: int = 16,
                 num_x_layers: int = 0,
                 num_R_layers: int = 0,
                 R2x_channels: int = 32,
                 out_channels: int = 2,
                 activation=nn.LeakyReLU(),
                 head_activation=nn.LeakyReLU(),
                 num_xx_head_layers: int = 2,
                 xx_head_hidden_channels: int | None = None,
                 xx_channels: int | None = None):
        super().__init__()
        assert num_x_layers >= 0 and num_R_layers >= 0

        # ---- x branch: (x->x)*num_x_layers ----
        self.x_branch = nn.ModuleList()
        cx = x_in_channels
        for _ in range(num_x_layers):
            out_c = xx_channels if (xx_channels is not None and _ == num_x_layers - 1) else x_hidden_channels
            self.x_branch.append(D6EquivxxLinear(cx, out_c))
            self.x_branch.append(D6EquivVectorBlockGating(act=activation))
            cx = out_c

        # ---- R branch: (R->R)*num_R_layers ----
        self.R_branch = nn.ModuleList()
        cR = R_in_channels
        for _ in range(num_R_layers):
            self.R_branch.append(D6EquivRRLinear(cR, R_hidden_channels))
            self.R_branch.append(D6EquivRRBlockGatingV2())
            cR = R_hidden_channels

        # ---- R->x
        self.R2x = D6LeftEquivRightInv_GroupConv(cR, R2x_channels, act=activation)

        # ---- x-head (x->x)
        fused_in = cx + R2x_channels
        if xx_head_hidden_channels is None:
            xx_head_hidden_channels = fused_in

        head_layers = []
        in_c = fused_in
        for _ in range(max(0, num_xx_head_layers - 1)):
            head_layers.append(D6EquivxxGroupConvDense(in_c, xx_head_hidden_channels,
                                                       activation=head_activation))
            in_c = xx_head_hidden_channels
        head_layers.append(D6EquivxxGroupConvDense(in_c, out_channels,
                                                   activation=head_activation))
        self.head = nn.Sequential(*head_layers)

        self.apply(_init_linear_weights)

    @staticmethod
    def _normalize_shapes(x: torch.Tensor, R: torch.Tensor):
        if x.dim() == 2 and x.size(-1) == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(-1) == 3, "x must be (B,Cx,3) or (B,3)"
        if R.dim() == 3 and R.size(-2) == 3 and R.size(-1) == 3:
            R = R.unsqueeze(1)
        assert R.dim() == 4 and R.size(-2) == 3 and R.size(-1) == 3, \
            "R must be (B,Cr,3,3) or (B,3,3)"
        return x, R

    def forward(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # x: (B, Cx, 3);  R: (B, CR, 3, 3)
        # x branch
        x, R = self._normalize_shapes(x, R)
        x_feat = x
        for layer in self.x_branch:
            x_feat = layer(x_feat)

        # R branch
        R_feat = R
        for layer in self.R_branch:
            R_feat = layer(R_feat)

        # R->x
        rx_feat = self.R2x(R_feat)  # (B, R2x_channels, 3)

        # fuse & head
        fused = torch.cat([x_feat, rx_feat], dim=1)  # (B, fused_in, 3)
        y = self.head(fused)                         # (B, out_channels, 3)
        return y


class TFENN_3_3(nn.Module):
    """
    x->x (0..L)  &  R->R (0..L)  &  R->x(GroupConv)  →  concat  →  x-head(GroupConv)
      x: (B, Cx, 3)
      R: (B, CR, 3, 3)
      y: (B, Cout, 3)
    """
    def __init__(self,
                 x_in_channels: int = 1,
                 R_in_channels: int = 1,
                 x_hidden_channels: int = 32,
                 R_hidden_channels: int = 16,
                 num_x_layers: int = 0,
                 num_R_layers: int = 0,
                 R2x_channels: int = 32,
                 out_channels: int = 2,
                 activation=nn.LeakyReLU(),
                 head_activation=nn.LeakyReLU(),
                 num_xx_head_layers: int = 2,
                 xx_head_hidden_channels: int | None = None,
                 xx_channels: int | None = None):
        super().__init__()
        assert num_x_layers >= 0 and num_R_layers >= 0

        # ---- x branch: (x->x)*num_x_layers ----
        self.x_branch = nn.ModuleList()
        cx = x_in_channels
        for _ in range(num_x_layers):
            out_c = xx_channels if (xx_channels is not None and _ == num_x_layers - 1) else x_hidden_channels
            self.x_branch.append(D6EquivxxLinear(cx, out_c))
            self.x_branch.append(D6EquivVectorBlockGating(act=activation))
            cx = out_c

        # ---- R branch: (R->R)*num_R_layers ----
        self.R_branch = nn.ModuleList()
        cR = R_in_channels
        for _ in range(num_R_layers):
            self.R_branch.append(D6EquivRRLinear(cR, R_hidden_channels))
            self.R_branch.append(D6EquivRR_SpectralPolyGate())
            cR = R_hidden_channels

        # ---- R->x:
        self.R2x = D6LeftEquivRightInv_GroupConv(cR, R2x_channels, act=activation)

        # ---- x-head (x->x):
        fused_in = cx + R2x_channels
        if xx_head_hidden_channels is None:
            xx_head_hidden_channels = fused_in

        head_layers = []
        in_c = fused_in
        for _ in range(max(0, num_xx_head_layers - 1)):
            head_layers.append(D6EquivxxGroupConvDense(in_c, xx_head_hidden_channels,
                                                       activation=head_activation))
            in_c = xx_head_hidden_channels
        head_layers.append(D6EquivxxGroupConvDense(in_c, out_channels,
                                                   activation=head_activation))
        self.head = nn.Sequential(*head_layers)

        self.apply(_init_linear_weights)

    @staticmethod
    def _normalize_shapes(x: torch.Tensor, R: torch.Tensor):
        if x.dim() == 2 and x.size(-1) == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(-1) == 3, "x must be (B,Cx,3) or (B,3)"
        if R.dim() == 3 and R.size(-2) == 3 and R.size(-1) == 3:
            R = R.unsqueeze(1)
        assert R.dim() == 4 and R.size(-2) == 3 and R.size(-1) == 3, \
            "R must be (B,Cr,3,3) or (B,3,3)"
        return x, R

    def forward(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # x: (B, Cx, 3);  R: (B, CR, 3, 3)
        # x branch
        x, R = self._normalize_shapes(x, R)
        x_feat = x
        for layer in self.x_branch:
            x_feat = layer(x_feat)

        # R branch
        R_feat = R
        for layer in self.R_branch:
            R_feat = layer(R_feat)

        # R->x
        rx_feat = self.R2x(R_feat)  # (B, R2x_channels, 3)

        # fuse & head
        fused = torch.cat([x_feat, rx_feat], dim=1)  # (B, fused_in, 3)
        y = self.head(fused)                         # (B, out_channels, 3)
        return y


class TFENN_3_4(nn.Module):
    def __init__(self,
                 x_in_channels: int = 1,
                 R_in_channels: int = 1,
                 x_hidden_channels: int = 32,
                 R_hidden_channels: int = 16,
                 num_x_layers: int = 0,
                 num_R_layers: int = 0,
                 R2x_channels: int = 32,
                 out_channels: int = 2,
                 activation=nn.LeakyReLU(),
                 head_activation=nn.LeakyReLU(),
                 num_xx_head_layers: int = 2,
                 xx_head_hidden_channels: int | None = None,
                 xx_channels: int | None = None):
        super().__init__()
        assert num_x_layers >= 0 and num_R_layers >= 0

        # ---- x branch: (x->x)*num_x_layers ----
        self.x_branch = nn.ModuleList()
        cx = x_in_channels
        for _ in range(num_x_layers):
            out_c = xx_channels if (xx_channels is not None and _ == num_x_layers - 1) else x_hidden_channels
            self.x_branch.append(D6EquivxxLinear(cx, out_c))
            self.x_branch.append(D6EquivVectorBlockGating(act=activation))
            cx = out_c

        # ---- R branch: (R->R)*num_R_layers ----
        self.R_branch = nn.ModuleList()
        cR = R_in_channels
        for _ in range(num_R_layers):
            self.R_branch.append(D6EquivRRLinear(cR, R_hidden_channels))
            self.R_branch.append(D6EquivRRBlockGating(act=nn.Identity()))
            cR = R_hidden_channels

        # ---- R->x:
        self.R2x = D6LeftEquivRightInv_GroupConv(cR, R2x_channels, act=activation)

        # ---- x-head (x->x):
        fused_in = cx + R2x_channels
        if xx_head_hidden_channels is None:
            xx_head_hidden_channels = fused_in

        head_layers = []
        in_c = fused_in
        for _ in range(max(0, num_xx_head_layers - 1)):
            head_layers.append(D6EquivxxGroupConvDense(in_c, xx_head_hidden_channels,
                                                       activation=head_activation))
            in_c = xx_head_hidden_channels
        head_layers.append(D6EquivxxGroupConvDense(in_c, out_channels,
                                                   activation=head_activation))
        self.head = nn.Sequential(*head_layers)

        self.apply(_init_linear_weights)

    @staticmethod
    def _normalize_shapes(x: torch.Tensor, R: torch.Tensor):
        if x.dim() == 2 and x.size(-1) == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(-1) == 3, "x must be (B,Cx,3) or (B,3)"
        if R.dim() == 3 and R.size(-2) == 3 and R.size(-1) == 3:
            R = R.unsqueeze(1)
        assert R.dim() == 4 and R.size(-2) == 3 and R.size(-1) == 3, \
            "R must be (B,Cr,3,3) or (B,3,3)"
        return x, R

    def forward(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # x: (B, Cx, 3);  R: (B, CR, 3, 3)
        # x branch
        x, R = self._normalize_shapes(x, R)
        x_feat = x
        for layer in self.x_branch:
            x_feat = layer(x_feat)

        # R branch
        R_feat = R
        for layer in self.R_branch:
            R_feat = layer(R_feat)

        # R->x
        rx_feat = self.R2x(R_feat)  # (B, R2x_channels, 3)

        # fuse & head
        fused = torch.cat([x_feat, rx_feat], dim=1)  # (B, fused_in, 3)
        y = self.head(fused)                         # (B, out_channels, 3)
        return y


class GroupConv1(nn.Module):
    def __init__(self,
                 x_in_channels: int = 1,
                 R_in_channels: int = 1,
                 x_hidden_channels: int = 32,
                 R_hidden_channels: int = 16,
                 num_x_layers: int = 0,
                 num_R_layers: int = 0,
                 R2x_channels: int = 32,
                 out_channels: int = 2,
                 x_activation=nn.LeakyReLU(),
                 R_activation=nn.LeakyReLU(),
                 head_activation=nn.LeakyReLU(),
                 num_xx_head_layers: int = 2,
                 xx_head_hidden_channels: int | None = None,
                 xx_channels: int | None = None):
        super().__init__()
        assert num_x_layers >= 0 and num_R_layers >= 0

        # ---- x branch: (x->x)*num_x_layers ----
        self.x_branch = nn.ModuleList()
        cx = x_in_channels
        for _ in range(num_x_layers):
            out_c = xx_channels if (xx_channels is not None and _ == num_x_layers - 1) else x_hidden_channels
            self.x_branch.append(D6EquivxxGroupConvDense(cx, out_c, activation=x_activation))
            cx = out_c

        # ---- R branch: (R->R)*num_R_layers ----
        self.R_branch = nn.ModuleList()
        cR = R_in_channels
        for _ in range(num_R_layers):
            self.R_branch.append(D6EquivRR_GroupConv(cR, R_hidden_channels, act=R_activation))
            cR = R_hidden_channels

        # ---- R->x:
        self.R2x = D6LeftEquivRightInv_GroupConv(cR, R2x_channels, act=R_activation)

        # ---- x-head (x->x):
        fused_in = cx + R2x_channels
        if xx_head_hidden_channels is None:
            xx_head_hidden_channels = fused_in

        head_layers = []
        in_c = fused_in
        for _ in range(max(0, num_xx_head_layers - 1)):
            head_layers.append(D6EquivxxGroupConvDense(in_c, xx_head_hidden_channels,
                                                       activation=head_activation))
            in_c = xx_head_hidden_channels
        head_layers.append(D6EquivxxGroupConvDense(in_c, out_channels,
                                                   activation=head_activation))
        self.head = nn.Sequential(*head_layers)

        self.apply(_init_linear_weights)

    @staticmethod
    def _normalize_shapes(x: torch.Tensor, R: torch.Tensor):
        if x.dim() == 2 and x.size(-1) == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(-1) == 3, "x must be (B,Cx,3) or (B,3)"
        if R.dim() == 3 and R.size(-2) == 3 and R.size(-1) == 3:
            R = R.unsqueeze(1)
        assert R.dim() == 4 and R.size(-2) == 3 and R.size(-1) == 3, \
            "R must be (B,Cr,3,3) or (B,3,3)"
        return x, R

    def forward(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        # x: (B, Cx, 3);  R: (B, CR, 3, 3)
        # x branch
        x, R = self._normalize_shapes(x, R)
        x_feat = x
        for layer in self.x_branch:
            x_feat = layer(x_feat)

        # R branch
        R_feat = R
        for layer in self.R_branch:
            R_feat = layer(R_feat)

        # R->x
        rx_feat = self.R2x(R_feat)  # (B, R2x_channels, 3)

        # fuse & head
        fused = torch.cat([x_feat, rx_feat], dim=1)  # (B, fused_in, 3)
        y = self.head(fused)                         # (B, out_channels, 3)
        return y


class GroupConvMLP(nn.Module):
    def __init__(
            self,
            hidden_dim:int=128,
            num_hidden_layers:int=2,
            act:nn.Module=nn.LeakyReLU()
    ):
        super().__init__()
        self.mlp=StandardMLP(
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            activation_fn=act
        )
        G = []
        Rx = torch.tensor([[1., 0., 0.],
                           [0., -1., 0.],
                           [0., 0., -1.]], dtype=torch.get_default_dtype())
        for k in range(6):
            th = k * math.pi / 3.0
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[c, -s, 0.],
                               [s, c, 0.],
                               [0., 0., 1.]], dtype=torch.get_default_dtype())
            G.append(Rz)
        for k in range(6):
            th = k * math.pi / 3.0
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[c, -s, 0.],
                               [s, c, 0.],
                               [0., 0., 1.]], dtype=torch.get_default_dtype())
            G.append(Rx @ Rz)
        self.register_buffer("G", torch.stack(G, dim=0))  # (12,3,3)

    @staticmethod
    def _normalize_shapes(x: torch.Tensor, R: torch.Tensor):
        if x.dim() == 2 and x.size(-1) == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 3 and x.size(-1) == 3, "x must be (B,Cx,3) or (B,3)"
        if R.dim() == 3 and R.size(-2) == 3 and R.size(-1) == 3:
            R = R.unsqueeze(1)
        assert R.dim() == 4 and R.size(-2) == 3 and R.size(-1) == 3, \
            "R must be (B,Cr,3,3) or (B,3,3)"
        return x, R

    def forward(self, x, R):
        x, R = self._normalize_shapes(x, R)
        G=self.G.to(dtype=x.dtype,device=x.device)
        B=x.size(0)
        y_sum=None
        for g1 in G:
            xg1 = torch.einsum('ij,bcj->bci', g1.T, x)
            Rg1 = torch.einsum('ij,bcjk->bcik', g1.T, R)
            for g2 in G:
                Rg2=torch.einsum('bcij,jk->bcik', Rg1, g2)
                z = torch.cat([Rg2.reshape(B,9),xg1.reshape(B,3)],dim=-1)
                z = self.mlp(z)
                yg = z.view(B,2,3)
                yg = torch.einsum('ij,bcj->bci',g1,yg)
                y_sum = yg if y_sum is None else (y_sum+yg)

        y=y_sum/(G.size(0)*G.size(0))
        return y


if __name__ == '__main__':
    import numpy as np
    from utils import axis_angle_to_matrix
    Rz60=axis_angle_to_matrix(torch.tensor([0.,0.,1.]),torch.pi/3)
    Rx180=axis_angle_to_matrix(torch.tensor([1.,0.,0.]),torch.pi)
    R1=Rz60@Rx180
    # R2=Rz60@axis_angle_to_matrix(torch.tensor([0.,0.,1.]),torch.pi/4)
    R2=axis_angle_to_matrix(torch.tensor([0.,0.,1.]),torch.pi/3)@axis_angle_to_matrix(torch.tensor([1.,0.1,0.]),torch.pi)

    # X_HIDDEN = 16  # channel: x-branch hidden channels
    # R_HIDDEN = 12  # channel: R-branch hidden channels
    # NUM_X_LAYERS = 2  # hidden_layer_num for x-branch
    # NUM_R_LAYERS = 2  # hidden_layer_num for R-branch
    # R2X_CHANNELS = 12  # r2xlayerchannel
    # NUM_XXGCONV_LAYERS = 2
    # XXGCONV_CHANNELS = None
    # XX_CHANNELS = 16
    # OUT_CHANNELS = 2
    #
    #
    # class Sigmoid3(nn.Module):
    #     def forward(self, x):
    #         return 3 * torch.sigmoid(x) - 3 / 2
    #
    # model = TFENN_3_3(x_hidden_channels=X_HIDDEN, num_x_layers=NUM_X_LAYERS, R_hidden_channels=R_HIDDEN,num_R_layers=NUM_R_LAYERS,
    #                 R2x_channels=R2X_CHANNELS, activation=Sigmoid3(), num_xx_head_layers=NUM_XXGCONV_LAYERS,xx_head_hidden_channels=XXGCONV_CHANNELS,
    #                 xx_channels=XX_CHANNELS,
    #                 head_activation=nn.LeakyReLU()
    #                 )
    # print(sum(p.numel() for p in model.parameters()))
    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # model = GroupConv1(x_hidden_channels=X_HIDDEN, num_x_layers=NUM_X_LAYERS, R_hidden_channels=R_HIDDEN,num_R_layers=NUM_R_LAYERS,
    #                 R2x_channels=R2X_CHANNELS, num_xx_head_layers=NUM_XXGCONV_LAYERS,xx_head_hidden_channels=XXGCONV_CHANNELS,
    #                 xx_channels=XX_CHANNELS,
    #                 head_activation=nn.LeakyReLU()
    #                 )
    model=GroupConvMLP(hidden_dim=108,num_hidden_layers=3)

    print(sum(p.numel() for p in model.parameters()))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    x = torch.tensor([[[3.,3.,1.]]])
    ax=torch.tensor([0.,0.,1.])
    angle=np.pi/2
    R = axis_angle_to_matrix(ax,angle)
    R = R.unsqueeze(0).unsqueeze(0)


    y1 = model(x,R)@R1

    y2 = model(x@R1, (R1.T)@R@R2)

    print(y1,y2)