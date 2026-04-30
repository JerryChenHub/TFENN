import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class D6EquivxxLinear(nn.Module):
    """
    D6-equivariant linear layer (x -> x), fixed axis = z.
    y = sum_i [ Theta[o,i,0] * (x_i P1) + Theta[o,i,1] * (x_i P2) ] No bias.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 2))

        self.register_buffer("P1", torch.diag(torch.tensor([1.0, 1.0, 0.0])))
        self.register_buffer("P2",  torch.diag(torch.tensor([0.0, 0.0, 1.0])))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Cin, 3) -> y: (B, Cout, 3)
        assert x.dim() == 3 and x.size(-1) == 3, "expected input shape (B, Cin, 3)"
        P1 = self.P1.to(dtype=x.dtype, device=x.device)
        P2 = self.P2.to(dtype=x.dtype, device=x.device)

        xP = torch.einsum('bci,ij->bcj', x, P1)
        xQ = torch.einsum('bci,ij->bcj', x, P2)
        comps = torch.stack([xP, xQ], dim=2)    # (B, Cin, 2, 3)

        y = torch.einsum('bisk,ois->bok', comps, self.weight)  # (B, Cout, 3)
        return y


class D6EquivVectorBlockGating(nn.Module):
    """
    D6-equivariant activation layer (R -> x) and (x->x), axis = z.
    Satisfies: for any Q in D6: f(Q^T x) = Q^T f(x).
    """
    def __init__(self, act: nn.Module = nn.Sigmoid(), eps: float = 1e-8):
        super().__init__()
        self.act = act
        self.eps = eps
        self.register_buffer("P1", torch.diag(torch.tensor([1.0, 1.0, 0.0])))
        self.register_buffer("P2", torch.diag(torch.tensor([0.0, 0.0, 1.0])))

    @torch.no_grad()
    def _check_shapes(self, y):
        assert y.dim() == 3 and y.size(-1) == 3, f"expected (B, C_out, 3), got {tuple(y.shape)}"

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        self._check_shapes(y)
        P1 = self.P1.to(dtype=y.dtype, device=y.device)
        P2 = self.P2.to(dtype=y.dtype, device=y.device)

        y1 = torch.einsum('ij,bcj->bci', P1, y)  # (B,C,3)
        y2 = torch.einsum('ij,bcj->bci', P2, y)  # (B,C,3)

        p1 = torch.linalg.norm(y1, dim=-1, keepdim=True) / math.sqrt(2.0)  # (B,C,1)
        p2 = torch.linalg.norm(y2, dim=-1, keepdim=True)                   # (B,C,1)

        # return y1 * self.act(p1)  + y2 * self.act(p2)
        return y1* self.act(p1) / (p1 + self.eps)+y2*self.act(p2) / (p2 + self.eps)

# class D6EquivVectorBlockGating(nn.Module):
#     """
#     Vector block gating with learnable gain/temp/bias; equivariant under D6.
#     """
#     def __init__(self, act: nn.Module = nn.Sigmoid(),
#                  gain_init: float = 2.0, temp_init: float = 1.0, bias_init: float = 0.0,
#                  eps: float = 1e-12):
#         super().__init__()
#         self.act = act
#         self.eps = eps
#         # projectors
#         self.register_buffer("Pxy", torch.diag(torch.tensor([1.0, 1.0, 0.0])))
#         self.register_buffer("Pz",  torch.diag(torch.tensor([0.0, 0.0, 1.0])))
#         # learnable scalars (positive via exp)
#         self.log_gain_xy = nn.Parameter(torch.log(torch.tensor(gain_init)))
#         self.log_gain_z  = nn.Parameter(torch.log(torch.tensor(gain_init)))
#         self.log_temp_xy = nn.Parameter(torch.log(torch.tensor(temp_init)))
#         self.log_temp_z  = nn.Parameter(torch.log(torch.tensor(temp_init)))
#         self.bias_xy     = nn.Parameter(torch.tensor(bias_init))
#         self.bias_z      = nn.Parameter(torch.tensor(bias_init))
#
#     def forward(self, y: torch.Tensor) -> torch.Tensor:
#         # y: (B, C, 3) or (B, 3); support broadcasting along channel
#         Pxy, Pz = self.Pxy, self.Pz
#         y_xy = y @ Pxy    # (..,3)
#         y_z  = y @ Pz
#
#         # norms with N (cardinality) normalization
#         N_xy = math.sqrt(2.0)
#         N_z  = 1.0
#         p_xy = torch.linalg.vector_norm(y_xy, dim=-1, keepdim=True).clamp_min(self.eps) / N_xy
#         p_z  = torch.linalg.vector_norm(y_z,  dim=-1, keepdim=True).clamp_min(self.eps) / N_z
#
#         gain_xy = torch.exp(self.log_gain_xy)
#         gain_z  = torch.exp(self.log_gain_z)
#         temp_xy = torch.exp(self.log_temp_xy)
#         temp_z  = torch.exp(self.log_temp_z)
#
#         g_xy = gain_xy * self.act(temp_xy * p_xy + self.bias_xy)
#         g_z  = gain_z  * self.act(temp_z  * p_z  + self.bias_z)
#
#         return y_xy * g_xy + y_z * g_z



class D6EquivxxGroupConvDense(nn.Module):
    """
    x->x 的 D6 左协变层（GroupConv；for 循环，无 *12 扩 batch）
    y(x) = (1/|D6|) Σ_g  R(g)^T · σ( W · vec(R(g) x) + b ) -> reshape为 (B,Cout,3)
    Activation Must be on
    """
    def __init__(self, in_channels: int, out_channels: int,
                 activation: nn.Module = nn.ELU(),
                 bias: bool = True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.act = activation
        self.inner = nn.Linear(in_channels * 3, out_channels * 3, bias=bias)
        nn.init.kaiming_uniform_(self.inner.weight, nonlinearity="leaky_relu")

        G = []
        Rx = torch.tensor([[1., 0., 0.],
                           [0.,-1., 0.],
                           [0., 0., -1.]], dtype=torch.get_default_dtype())
        for k in range(6):
            th = k * math.pi / 3.0
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[ c, -s, 0.],
                               [ s,  c, 0.],
                               [0., 0., 1.]], dtype=torch.get_default_dtype())
            G.append(Rz)
        for k in range(6):
            th = k * math.pi / 3.0
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[ c, -s, 0.],
                               [ s,  c, 0.],
                               [0., 0., 1.]], dtype=torch.get_default_dtype())
            G.append(Rx @ Rz)
        self.register_buffer("G", torch.stack(G, dim=0))  # (12,3,3)

    @torch.no_grad()
    def _check(self, x: torch.Tensor):
        assert x.dim() == 3 and x.size(-1) == 3, \
            f"x must be (B, Cin, 3), got {tuple(x.shape)}"
        assert x.size(1) == self.in_channels, \
            f"in_channels mismatch: {x.size(1)} vs {self.in_channels}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, Cin, 3) -> y: (B, Cout, 3)
        """
        self._check(x)
        B, Cin = x.size(0), x.size(1)
        G = self.G.to(device=x.device, dtype=x.dtype)

        y_sum = None
        for g in G:
            xg = torch.einsum('ij,bcj->bci', g, x)             # (B,Cin,3)

            z  = xg.reshape(B, Cin * 3)
            z  = self.inner(z)                                 # (B, Cout*3)
            z  = self.act(z)
            yg = z.view(B, self.out_channels, 3)               # (B,Cout,3)

            yg = torch.einsum('ji,bcj->bci', g, yg)            # (B,Cout,3)
            y_sum = yg if y_sum is None else (y_sum + yg)

        y_sum = y_sum / G.size(0)
        return y_sum


class D6EquivRR_GroupConv(nn.Module):
    """
    Strict D6 bi-equivariant layer (R -> R) via double group convolution.

    L(R) = (1/|D6|^2) sum_{g1,g2}  g1 · Phi( g1^T · R · g2 ) · g2^T
    =>  L(Q1^T R Q2) = Q1^T L(R) Q2
    Input : x ∈ (B, C_in, 3, 3)
    Output: y ∈ (B, C_out, 3, 3)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 act: nn.Module = nn.LeakyReLU(), bias: bool = True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.act = act

        # shared MLP on vec(3x3) per channel
        self.mlp = nn.Linear(in_channels * 9, out_channels * 9, bias=bias)
        nn.init.kaiming_uniform_(self.mlp.weight, nonlinearity="leaky_relu")

        # D6 = { Rz(k*60°), Rx(180°) Rz(k*60°) }
        G = []
        Rx180 = torch.tensor([[1., 0., 0.],
                              [0.,-1., 0.],
                              [0., 0.,-1.]], dtype=torch.get_default_dtype())
        for k in range(6):
            th = k * math.pi / 3.0
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[ c, -s, 0.],
                               [ s,  c, 0.],
                               [0., 0., 1.]], dtype=torch.get_default_dtype())
            G.append(Rz)
        for k in range(6):
            th = k * math.pi / 3.0
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[ c, -s, 0.],
                               [ s,  c, 0.],
                               [0., 0., 1.]], dtype=torch.get_default_dtype())
            G.append(Rx180 @ Rz)
        self.register_buffer("G", torch.stack(G, dim=0))  # (12,3,3)

    @torch.no_grad()
    def _check(self, x: torch.Tensor):
        assert x.dim() == 4 and x.size(-2) == 3 and x.size(-1) == 3, \
            f"expected (B, C_in, 3, 3), got {tuple(x.shape)}"
        assert x.size(1) == self.in_channels, \
            f"C_in mismatch: layer={self.in_channels}, x={x.size(1)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        B, Cin = x.size(0), x.size(1)
        G = self.G.to(dtype=x.dtype, device=x.device)  # (12,3,3)

        y_sum = None
        # double sum over g1 (left), g2 (right)
        for g1 in G:
            # left action: g1^T @ x
            x_l = torch.einsum('ij,bcjk->bcik', g1.T, x)
            for g2 in G:
                # right action: (. ) @ g2
                x_lr = torch.einsum('bcij,jk->bcik', x_l, g2)        # (B,Cin,3,3)

                z = x_lr.reshape(B, Cin * 9)
                z = self.mlp(z)                                      # (B, Cout*9)
                z = self.act(z)
                y_hat = z.view(B, self.out_channels, 3, 3)           # (B,Cout,3,3)

                # undo: left by g1, right by g2^T  (== multiply by inverses)
                y_hat = torch.einsum('ij,bcjk->bcik', g1, y_hat)
                y_hat = torch.einsum('bcij,jk->bcik', y_hat, g2.T)

                y_sum = y_hat if y_sum is None else (y_sum + y_hat)

        y = y_sum / (G.size(0) * G.size(0))
        return y



class D6EquivRRLinear(nn.Module):
    """
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 4))

        self.register_buffer("P1", torch.diag(torch.tensor([1.0, 1.0, 0.0])))
        self.register_buffer("P2", torch.diag(torch.tensor([0.0, 0.0, 1.0])))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    @torch.no_grad()
    def _check_shapes(self, x):
        assert x.dim() == 4 and x.size(-1) == 3 and x.size(-2) == 3, \
            f"expected (B, C_in, 3, 3), got {tuple(x.shape)}"
        assert x.size(1) == self.in_channels, \
            f"C_in mismatch: layer={self.in_channels}, x={x.size(1)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, 3, 3)
        self._check_shapes(x)

        P1 = self.P1.to(dtype=x.dtype, device=x.device)
        P2 = self.P2.to(dtype=x.dtype, device=x.device)

        x_P1   = torch.einsum('bcij,jk->bcik', x, P1)   # x @ Pp
        x_P2   = torch.einsum('bcij,jk->bcik', x, P2)   # x @ Pz
        block_11 = torch.einsum('ij,bcjk->bcik', P1, x_P1)  # Pp x Pp
        block_12 = torch.einsum('ij,bcjk->bcik', P1, x_P2)  # Pp x Pz
        block_21 = torch.einsum('ij,bcjk->bcik', P2, x_P1)  # Pz x Pp
        block_22 = torch.einsum('ij,bcjk->bcik', P2, x_P2)  # Pz x Pz

        #(B, C_in, 4, 3, 3)
        blocks = torch.stack([block_11, block_12, block_21, block_22], dim=2)

        #w ∈ (C_out, C_in, 4)
        #(B, C_out, 3, 3)
        y = torch.einsum('bcfij,ocf->boij', blocks, self.weight)
        return y


class D6EquivRRBlockGatingTH(nn.Module):
    """
    Strictly D6 bi-equivariant R->R activation.
    x: (B, C, 3, 3)
    f(X) = sum_k phi_k(||B_k||/kappa_k) * B_k,  B_k in {P1XP1, P1XP2, P2XP1, P2XP2}
    """
    def __init__(self, eps: float = 1e-8, use_cardinality_scale: bool = True):
        super().__init__()
        self.eps = eps

        self.register_buffer("P1", torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)))
        self.register_buffer("P2", torch.diag(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)))

        self.register_buffer(
            "kappa",
            torch.tensor([2.0, math.sqrt(2.0), math.sqrt(2.0), 1.0], dtype=torch.float32)
        )
        self.use_cardinality_scale = use_cardinality_scale

        #phi_k(r) = 1 + gamma_k * tanh(alpha_k * r + beta_k)
        self.alpha = nn.Parameter(torch.zeros(4))
        self.beta  = nn.Parameter(torch.zeros(4))
        self.gamma = nn.Parameter(torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.size(-2) == 3 and x.size(-1) == 3, "x must be (B,C,3,3)"
        P1 = self.P1.to(dtype=x.dtype, device=x.device)
        P2 = self.P2.to(dtype=x.dtype, device=x.device)

        xP1 = torch.einsum('bcij,jk->bcik', x, P1)
        xP2 = torch.einsum('bcij,jk->bcik', x, P2)
        b11 = torch.einsum('ij,bcjk->bcik', P1, xP1)
        b12 = torch.einsum('ij,bcjk->bcik', P1, xP2)
        b21 = torch.einsum('ij,bcjk->bcik', P2, xP1)
        b22 = torch.einsum('ij,bcjk->bcik', P2, xP2)
        blocks = torch.stack([b11, b12, b21, b22], dim=2)  # (B,C,4,3,3)

        norms = blocks.square().sum(dim=(-1, -2), keepdim=True).sqrt()  # (B,C,4,1,1)

        if self.use_cardinality_scale:
            kappa = self.kappa.to(dtype=x.dtype, device=x.device).view(1,1,4,1,1).clamp_min(self.eps)
            r = norms / kappa
        else:
            r = norms

        alpha = self.alpha.to(dtype=x.dtype, device=x.device).view(1,1,4,1,1)
        beta  = self.beta .to(dtype=x.dtype, device=x.device).view(1,1,4,1,1)
        gamma = self.gamma.to(dtype=x.dtype, device=x.device).view(1,1,4,1,1)

        phi = 1.0 + gamma * torch.tanh(alpha * r + beta)  # (B,C,4,1,1)

        y_blocks = phi * blocks
        y = y_blocks.sum(dim=2)  # (B,C,3,3)
        return y


class D6EquivRRBlockGatingV2(nn.Module):
    """
    Strict D6 bi-equivariant R->R activation with learnable, non-zero-grad identity init.
    x: (B, C, 3, 3)
    y = x + eta * ( sum_k phi_k(||B_k||/kappa_k) * B_k - x )
    """
    def __init__(self, eps: float = 1e-8,
                 use_cardinality_scale: bool = True,
                 init_eta: float = 0.05,
                 cap_phi: float | None = None):
        super().__init__()
        self.eps = eps
        self.use_cardinality_scale = use_cardinality_scale
        self.cap_phi = cap_phi

        self.register_buffer("P1", torch.diag(torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)))
        self.register_buffer("P2", torch.diag(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)))

        self.register_buffer(
            "kappa",
            torch.tensor([2.0, math.sqrt(2.0), math.sqrt(2.0), 1.0], dtype=torch.float32)
        )

        self.a = nn.Parameter(torch.zeros(4))
        self.b = nn.Parameter(torch.zeros(4))

        logit = math.log(init_eta) - math.log(1 - init_eta)
        self.logit_eta = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.size(-2) == 3 and x.size(-1) == 3, "x must be (B,C,3,3)"
        P1 = self.P1.to(dtype=x.dtype, device=x.device)
        P2 = self.P2.to(dtype=x.dtype, device=x.device)

        xP1 = torch.einsum('bcij,jk->bcik', x, P1)
        xP2 = torch.einsum('bcij,jk->bcik', x, P2)
        b11 = torch.einsum('ij,bcjk->bcik', P1, xP1)
        b12 = torch.einsum('ij,bcjk->bcik', P1, xP2)
        b21 = torch.einsum('ij,bcjk->bcik', P2, xP1)
        b22 = torch.einsum('ij,bcjk->bcik', P2, xP2)
        blocks = torch.stack([b11, b12, b21, b22], dim=2)  # (B,C,4,3,3)

        norms = blocks.square().sum(dim=(-1, -2), keepdim=True).sqrt()  # (B,C,4,1,1)
        if self.use_cardinality_scale:
            kappa = self.kappa.to(dtype=x.dtype, device=x.device).view(1,1,4,1,1).clamp_min(self.eps)
            r = norms / kappa
        else:
            r = norms

        a = self.a.to(dtype=x.dtype, device=x.device).view(1,1,4,1,1)
        b = self.b.to(dtype=x.dtype, device=x.device).view(1,1,4,1,1)
        denom = F.softplus(b) + self.eps
        phi = F.softplus(a * r + b) / denom           # >= 0

        if self.cap_phi is not None:
            m = float(self.cap_phi) - 1.0
            d = phi - 1.0
            phi = 1.0 + m * torch.tanh(d / (m + 1e-12))

        y_bar = (phi * blocks).sum(dim=2)             # (B,C,3,3)
        eta = torch.sigmoid(self.logit_eta).to(dtype=x.dtype, device=x.device)  # (0,1)
        y = x + eta * (y_bar - x)
        return y


class D6EquivRR_SpectralPolyGate(nn.Module):
    """
    D6 bi-equivariant R->R activation via spectral polynomial gate (SVD-free).
    对每个块 B∈{P1XP1, P1XP2, P2XP1, P2XP2} 做:
        f(B) = sum_{n=0}^K w_{k,n} * B * (B^T B)^n
    """
    def __init__(self, K: int = 2, init_eta: float = 0.1, eps: float = 1e-8):
        super().__init__()
        assert K >= 1
        self.K = K
        self.eps = eps

        self.register_buffer("P1", torch.diag(torch.tensor([1., 1., 0.], dtype=torch.float32)))
        self.register_buffer("P2", torch.diag(torch.tensor([0., 0., 1.], dtype=torch.float32)))

        self.w_rest = nn.Parameter(torch.zeros(4, K))

        logit = math.log(init_eta) - math.log(1 - init_eta)
        self.logit_eta = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    def _spectral_poly(self, B: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        B: (..., 3, 3)
        w: (K,)  —— n=1..K  n=0  K=1
        sum_{n=0}^K w_n * B * (B^T B)^n
        """
        BtB = B.transpose(-1, -2) @ B                      # (..., 3, 3)
        Y = B.clone()                                      # n=0 coef=1
        term = B
        # for (B^T B)^n：term = B * (BtB)^n
        for n in range(1, self.K + 1):
            term = term @ BtB                              # B*(BtB)^n
            coef = w[..., n - 1].view(*w.shape[:-1], 1, 1)
            Y = Y + coef * term
        return Y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.size(-2) == 3 and x.size(-1) == 3, "x must be (B,C,3,3)"
        P1 = self.P1.to(dtype=x.dtype, device=x.device)
        P2 = self.P2.to(dtype=x.dtype, device=x.device)

        xP1 = torch.einsum('bcij,jk->bcik', x, P1)
        xP2 = torch.einsum('bcij,jk->bcik', x, P2)
        B11 = torch.einsum('ij,bcjk->bcik', P1, xP1)
        B12 = torch.einsum('ij,bcjk->bcik', P1, xP2)
        B21 = torch.einsum('ij,bcjk->bcik', P2, xP1)
        B22 = torch.einsum('ij,bcjk->bcik', P2, xP2)
        blocks = [B11, B12, B21, B22]

        Ys = []
        for k, Bk in enumerate(blocks):
            wk = self.w_rest[k]  # (K,)
            Ys.append(self._spectral_poly(Bk, wk))
        y_bar = sum(Ys)  # (B,C,3,3)

        eta = torch.sigmoid(self.logit_eta).to(dtype=x.dtype, device=x.device)  # (0,1)
        y = x + eta * (y_bar - x)
        return y


class D6EquivRRBlockGating(nn.Module):
    """
    D6-equivariant R->R activation
    """
    def __init__(self, act: nn.Module = nn.SiLU(), eps: float = 1e-8):
        super().__init__()
        self.act = act
        self.eps = eps
        self.register_buffer("P1", torch.diag(torch.tensor([1.0, 1.0, 0.0])))
        self.register_buffer("P2", torch.diag(torch.tensor([0.0, 0.0, 1.0])))
        self.register_buffer("cardinality_", torch.tensor([2.0, np.sqrt(2), np.sqrt(2), 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3, 3)
        assert x.dim() == 4 and x.size(-1) == 3 and x.size(-2) == 3

        P1 = self.P1.to(dtype=x.dtype, device=x.device)
        P2 = self.P2.to(dtype=x.dtype, device=x.device)
        N  = self.cardinality_.to(dtype=x.dtype, device=x.device).view(1, 1, 4, 1, 1)

        xP1 = torch.einsum('bcij,jk->bcik', x, P1)
        xP2 = torch.einsum('bcij,jk->bcik', x, P2)
        b11 = torch.einsum('ij,bcjk->bcik', P1, xP1)
        b12 = torch.einsum('ij,bcjk->bcik', P1, xP2)
        b21 = torch.einsum('ij,bcjk->bcik', P2, xP1)
        b22 = torch.einsum('ij,bcjk->bcik', P2, xP2)
        blocks = torch.stack([b11, b12, b21, b22], dim=2)  # (B,C,4,3,3)

        norms = (blocks.pow(2).sum(dim=(-1, -2), keepdim=True)).sqrt().clamp_min(self.eps)  # (B,C,4,1,1)
        gates = self.act(norms/N)
        # gates = self.act(norms)
        y_blocks = gates/(norms).clamp_min(self.eps) * blocks
        y = y_blocks.sum(dim=2)  # (B,C,3,3)
        return y


class D6EquivRR_SVDActivation(nn.Module):
    """
    D6-equivariant R->R activation via SVD:
      For X ∈ R^{3x3}, compute X = U diag(s) Vh, then
      phi(X) = U diag(act(s_clamped)) Vh
    Equivariance: phi(Q1^T X Q2) = Q1^T phi(X) Q2 for any orthogonal Q1,Q2 in the group.
    """
    def __init__(self, act: nn.Module = nn.Sigmoid(), eps: float = 1e-8):
        super().__init__()
        self.act = act
        self.eps = eps

    @torch.no_grad()
    def _check_shapes(self, x: torch.Tensor):
        assert x.dim() == 4 and x.size(-1) == 3 and x.size(-2) == 3, \
            f"expected (B, C, 3, 3), got {tuple(x.shape)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3, 3)
        self._check_shapes(x)
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)  # shapes: (B,C,3,3), (B,C,3), (B,C,3,3)
        S_act = self.act(S.clamp_min(self.eps))
        y = U @ torch.diag_embed(S_act) @ Vh                 # (B,C,3,3)
        return y


class D6EquivRxLinear(nn.Module):
    """
    D6-equivariant linear layer (R -> x), axis = z. Left-covariant only!!!!!
    Satisfies: for any Q in D6: f(Q^T R) = Q^T f(R).
    (No right-invariance is imposed.)

    Basis (6 filters; per column k in {x,y,z}):
      vP_k(R) = P_perp @ (R @ e_k)               # planar vector of column k
      vZ_k(R) = P_para @ (R @ e_k)               # axial vector from column k (one shot, no ez)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels

        # weight: (C_out, C_in, 3 columns (k), 2 comps (t: planar/axial))
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 3, 2))

        self.register_buffer("P1", torch.diag(torch.tensor([1.0, 1.0, 0.0])))  # diag(1,1,0)
        self.register_buffer("P2", torch.diag(torch.tensor([0.0, 0.0, 1.0])))  # diag(0,0,1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    @torch.no_grad()
    def _check_shapes(self, x):
        # x: (B, C_in, 3, 3)
        assert x.dim() == 4 and x.size(-1) == 3 and x.size(-2) == 3, \
            f"expected (B, C_in, 3, 3), got {tuple(x.shape)}"
        assert x.size(1) == self.in_channels, \
            f"C_in mismatch: layer={self.in_channels}, x={x.size(1)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, 3, 3) -> y: (B, C_out, 3)
        self._check_shapes(x)

        P1 = self.P1.to(dtype=x.dtype, device=x.device)
        P2 = self.P2.to(dtype=x.dtype, device=x.device)

        vP_all = torch.einsum('ij,bcjk->bcik', P1, x)  # (B, C, i, k) planar
        vZ_all = torch.einsum('ij,bcjk->bcik', P2, x)  # (B, C, i, k) axial (single step via projector)

        # stack along type axis t∈{0:planar, 1:axial}; get (B, C, i, k, t)
        comps = torch.stack([vP_all, vZ_all], dim=-1)  # (B, C, 3, 3, 2)

        # weight: (O, C, k, t). Contract over C,k,t; keep (B,O,i).
        y = torch.einsum('bcikt,ockt->boi', comps, self.weight)  # (B, C_out, 3)
        return y


class D6EquivRxDense(nn.Module):
    """
    D6-equivariant Dense: Linear (R->x) + equivariant activation. No bias!
    """
    def __init__(self, in_channels: int, out_channels: int, act: nn.Module = nn.Sigmoid(), eps: float = 1e-8):
        super().__init__()
        self.lin = D6EquivRxLinear(in_channels, out_channels)
        self.act = D6EquivVectorBlockGating(act=act, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        y = self.act(y)
        return y


class D6LeftEquivRightInv_TFENNGroupConv(nn.Module):
    """
    Left-equivariant & right-invariant layer
    Left-equivariant achieved by TFENN RxDense,
    GroupConv on right side to resolve the invariant.
    f(R(g1)^T R R(g2)) = R(g1)^T f(R).
    x ∈ (B, C_in, 3, 3)
    y ∈ (B, C_out, 3)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 act: nn.Module = nn.Sigmoid()):
        super().__init__()
        self.L = D6EquivRxDense(in_channels, out_channels, act=act)

        G = []
        R_x180 = torch.tensor([[1., 0., 0.],
                          [0., -1., 0.],
                          [0., 0., -1.]], dtype=torch.get_default_dtype())
        for k in range(6):
            theta = k * math.pi / 3.0
            c, s = math.cos(theta), math.sin(theta)
            Rz = torch.tensor([[ c, -s, 0.],
                               [ s,  c, 0.],
                               [0., 0., 1.]], dtype=torch.get_default_dtype())
            G.append(Rz)
        for k in range(6):
            theta = k * math.pi / 3.0
            c, s = math.cos(theta), math.sin(theta)
            Rz = torch.tensor([[ c, -s, 0.],
                               [ s,  c, 0.],
                               [0., 0., 1.]], dtype=torch.get_default_dtype())
            G.append(R_x180 @ Rz)
        self.register_buffer("G2", torch.stack(G, dim=0))  # (12,3,3)

    @torch.no_grad()
    def _check_shapes(self, x: torch.Tensor):
        assert x.dim() == 4 and x.size(-1) == 3 and x.size(-2) == 3, \
            f"expected (B, C_in, 3, 3), got {tuple(x.shape)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_shapes(x)
        G = self.G2.to(dtype=x.dtype, device=x.device)   # (12,3,3)
        y_sum = None
        for g in G:
            xg = torch.einsum('bcij,jk->bcik', x, g)
            yg = self.L(xg)
            y_sum = yg if y_sum is None else (y_sum + yg)

        y = y_sum / G.size(0)
        return y


class D6LeftEquivRightInv_GroupConv(nn.Module):
    """
    Left-equivariant & right-invariant via full group convolution:
      f(R) = (1/|G|^2) sum_{g1,g2 in D6}  g1^T · MLP( g1^T · R · g2 )
    Input : x ∈ (B, C_in, 3, 3)
    Output: y ∈ (B, C_out, 3)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 act: nn.Module = nn.Sigmoid(),
                 bias: bool = True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.act = act
        # shared MLP for all transformed inputs
        self.mlp = nn.Linear(in_channels * 9, out_channels * 3, bias=bias)
        nn.init.kaiming_uniform_(self.mlp.weight, nonlinearity="leaky_relu")

        # build D6 = { Rz(k*60°), Rx(180°) Rz(k*60°) }
        G = []
        Rx180 = torch.tensor([[1., 0., 0.],
                              [0.,-1., 0.],
                              [0., 0.,-1.]], dtype=torch.get_default_dtype())
        for k in range(6):
            th = k * math.pi / 3.0
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[ c, -s, 0.],
                               [ s,  c, 0.],
                               [0., 0., 1.]], dtype=torch.get_default_dtype())
            G.append(Rz)
        for k in range(6):
            th = k * math.pi / 3.0
            c, s = math.cos(th), math.sin(th)
            Rz = torch.tensor([[ c, -s, 0.],
                               [ s,  c, 0.],
                               [0., 0., 1.]], dtype=torch.get_default_dtype())
            G.append(Rx180 @ Rz)
        self.register_buffer("G", torch.stack(G, dim=0))  # (12,3,3)

    @torch.no_grad()
    def _check(self, x: torch.Tensor):
        assert x.dim() == 4 and x.size(-2) == 3 and x.size(-1) == 3, \
            f"expected (B, C_in, 3, 3), got {tuple(x.shape)}"
        assert x.size(1) == self.in_channels, \
            f"C_in mismatch: layer={self.in_channels}, x={x.size(1)}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        B, Cin = x.size(0), x.size(1)
        G = self.G.to(dtype=x.dtype, device=x.device)  # (12,3,3)

        y_sum = None
        # double sum over g1 (left), g2 (right)
        for g1 in G:
            # left action: g1^T @ x
            x_l = torch.einsum('ij,bcjk->bcik', g1.T, x)
            for g2 in G:
                # right action: (.. ) @ g2
                x_lr = torch.einsum('bcij,jk->bcik', x_l, g2)          # (B,Cin,3,3)
                z = x_lr.reshape(B, Cin * 9)
                z = self.mlp(z)                                        # (B, Cout*3)
                z = self.act(z)
                yg = z.view(B, self.out_channels, 3)                   # (B,Cout,3)
                # undo left action on output for left-equivariance
                yg = torch.einsum('ij,bcj->bci', g1, yg)               # g1 · y
                y_sum = yg if y_sum is None else (y_sum + yg)

        y = y_sum / (G.size(0) * G.size(0))
        return y






if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)


    L=D6LeftEquivRightInv_GroupConv(in_channels=32,out_channels=32)
    print(sum(p.numel() for p in L.parameters()))
    exit()


    from utils import axis_angle_to_matrix
    # R60_z=axis_angle_to_matrix(torch.tensor([0,0.,1.]),torch.pi/3)
    # R180_x=axis_angle_to_matrix(torch.tensor([1,0,0.]),torch.pi)
    # x=torch.tensor([[[-3.,3.,1.]]])



    #Check the activation
    # act=D6EquivVectorBlockGating(act=nn.Sigmoid())
    # y1=act(x@R60_z@R180_x)
    # y2=act(x)@R60_z@R180_x
    # print((y1,y2))
    # print(torch.max(torch.abs(y1-y2)).item())

    # L=D6EquivxxGroupConvDense(in_channels=1, out_channels=2,activation=nn.Identity())
    # y1=L(x@R60_z@R180_x)
    # y2=L(x)@R60_z@R180_x
    # print((y1,y2))

    # layer = D6EquivRRLinear(in_channels=2, out_channels=4)
    # Q1=axis_angle_to_matrix(torch.tensor([0,0,1.]),torch.pi/12)
    # Q2=axis_angle_to_matrix(torch.tensor([1.,0.,0.]),torch.pi)
    # with torch.no_grad():
    #     x = torch.randn(5, 2, 3, 3)
    #     x1 = torch.einsum('ij,bcjk->bcik', Q1.T, x)
    #     x1 = torch.einsum('bcij,jk->bcik', x1, Q2)
    #     y1 = layer(x1)
    #     y = layer(x)
    #     y2 = torch.einsum('ij,bcjk->bcik', Q1.T, y)
    #     y2 = torch.einsum('bcij,jk->bcik', y2, Q2)
    #     print((y1 - y2).abs().max().item())

    # layer = D6EquivRRBlockGating()
    # Q1=axis_angle_to_matrix(torch.tensor([0,0.,1.]),torch.pi/12)
    # Q2=axis_angle_to_matrix(torch.tensor([1.,0.,0.]),torch.pi)
    # with torch.no_grad():
    #     x = torch.randn(5, 2, 3, 3)
    #     x1 = torch.einsum('ij,bcjk->bcik', Q1.T, x)
    #     x1 = torch.einsum('bcij,jk->bcik', x1, Q2)
    #     y1 = layer(x1)
    #     y = layer(x)
    #     y2 = torch.einsum('ij,bcjk->bcik', Q1.T, y)
    #     y2 = torch.einsum('bcij,jk->bcik', y2, Q2)
    #     print((y1 - y2).abs().max().item())


    # layer = D6EquivRxLinear(in_channels=2, out_channels=3)
    # x = torch.randn(5, 2, 3, 3)
    # Qz = axis_angle_to_matrix(torch.tensor([0.,0.,1.]), torch.pi/3)
    # Qx = axis_angle_to_matrix(torch.tensor([1.,0.,0.]),torch.pi)
    # Q1=Qx
    # Q2=Qz
    # with torch.no_grad():
    #     x_t = torch.einsum('ij,bcjk->bcik', Q1.T, x)     # left action
    #     x_t = torch.einsum('bcij,jk->bcik', x_t, Q2)     # right action
    #     y1  = layer(x_t)
    #     y   = layer(x)
    #     y2  = torch.einsum('ij,bcj->bci', Q1.T, y)       # expected transform
    #     print("max equivariance error:", (y1 - y2).abs().max().item())

    # layer = D6EquivRxLinear(in_channels=2, out_channels=3)
    # layer = D6EquivRxDense(in_channels=2, out_channels=3)
    # x = torch.randn(5, 2, 3, 3)
    # Qz = axis_angle_to_matrix(torch.tensor([0.,0.,1.]), torch.pi/17)
    # Qx = axis_angle_to_matrix(torch.tensor([1.,0.,0.]),torch.pi)
    # Q1=Qx@Qz
    # Q2=torch.diag(torch.tensor([1.,1.,1.]))
    # with torch.no_grad():
    #     x_t = torch.einsum('ij,bcjk->bcik', Q1.T, x)     # left action
    #     x_t = torch.einsum('bcij,jk->bcik', x_t, Q2)     # right action
    #     y1  = layer(x_t)
    #     y   = layer(x)
    #     y2  = torch.einsum('ij,bcj->bci', Q1.T, y)       # expected transform
    #     y2 = torch.einsum('bcj,jk->bck', y2, Q2)
    #     print("max equivariance error:", (y1 - y2).abs().max().item())

    layer= D6LeftEquivRightInv_GroupConv(in_channels=2, out_channels=3, act=nn.Sigmoid())
    x = torch.randn(5, 2, 3, 3)
    Qz = axis_angle_to_matrix(torch.tensor([0., 0., 1.]), torch.pi / 3)
    Qx = axis_angle_to_matrix(torch.tensor([1.,0.,0.]),torch.pi)

    Q1 = axis_angle_to_matrix(torch.tensor([0., 0., 1.]), torch.pi / 3 )
    Q2 = axis_angle_to_matrix(torch.tensor([1., 0., 0.]), torch.pi)
    g1 = Qz
    g2 = Q2



    x_t = torch.einsum('ij,bcjk->bcik', g1.T, x)  # left: g1^T @ x
    x_t = torch.einsum('bcij,jk->bcik', x_t, g2)  # right: (..) @ g2

    y1 = layer(x_t)  # f(g1^T x g2)

    y = layer(x)  # f(x)
    y2 = torch.einsum('ij,bcj->bci', g1.T, y)  # 期望: g1^T f(x)   (不对输出右乘 g2)

    print((y1 - y2).abs().max().item())


