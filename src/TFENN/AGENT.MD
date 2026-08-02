# A/B Finite-Group Model: Mathematical Contract

本文使用 **STF**（symmetric trace-free）。所有实现必须遵守以下顺序与变换律。

## 1. 坐标与群作用

$G\subset SO(3)$ 为分子的 proper point group；$R_i$ 将 body frame 旋转到 world frame。边 $i\leftarrow j$ 表示 sender $j$ 到 receiver $i$。

\[
x'_{\pi(i)}=h x_i+t,\qquad R'_{\pi(i)}=hR_i g_i,\qquad g_i\in G.
\]

receiver-local geometry：

\[
X_{ij}=R_i^\top(x_j-x_i),\qquad Q_{ij}=R_i^\top R_j,
\]

\[
X'_{\pi(i)\pi(j)}=g_i^\top X_{ij},\qquad
Q'_{\pi(i)\pi(j)}=g_i^\top Q_{ij}g_j.
\]

## 2. STF 数学层

\[
V_\ell=\operatorname{Sym}_0^\ell(\mathbb R^3),\qquad \dim V_\ell=2\ell+1.
\]

canonical STF basis $B_\ell$ 满足

\[
T_\ell B_\ell=0,\qquad B_\ell^\top B_\ell=I.
\]

旋转表示：

\[
D_\ell(R)=B_\ell^\top\operatorname{Sym}^\ell(R)B_\ell,
\]

\[
D_\ell(RS)=D_\ell(R)D_\ell(S),\qquad D_\ell(R)^\top D_\ell(R)=I.
\]

$B_\ell$ 只能从唯一的 CPU-float64 master basis cast；不得按 dtype/device 重新求基。

STF coupling 必须满足

\[
C_{\ell_1\ell_2\to L}(D_{\ell_1}(R)a,D_{\ell_2}(R)b)
=D_L(R)C_{\ell_1\ell_2\to L}(a,b).
\]

coupling 离线生成，运行时仅做 batched contraction。

## 3. A/B 通道与姿态

\[
A=V_1\simeq\mathbb R^3,\qquad
B=\bigoplus_{\ell\in\mathcal L}V_\ell^{\oplus m_\ell}.
\]

benzene 的 primitive ranks 为 $\mathcal L=\{2,6\}$。anchor 满足

\[
D_\ell(g)K_{\ell,a}=K_{\ell,a}.
\]

边姿态编码：

\[
Z_{ij}^{(\ell,a)}=D_\ell(Q_{ij})K_{\ell,a},
\]

\[
Z_\ell(g_i^\top Q_{ij}g_j)=D_\ell(g_i^\top)Z_\ell(Q_{ij}).
\]

因此 sender gauge $g_j$ 被消除，A/B edge features 只按 receiver $g_i$ 协变。

## 4. AA / AB / BA / BB

所有线性 map 满足 intertwiner 约束

\[
M\rho_{\rm in}(g)=\rho_{\rm out}(g)M.
\]

| 名称 | map |
|---|---|
| AA | $A\to A$ |
| AB | $A\to B$ |
| BA | $B\to A$ |
| BB | $B\to B$ |

typed update：

\[
A'=AA(A)+BA(B)+P_{\to A}(A,B),
\]

\[
B'=AB(A)+BB(B)+P_{\to B}(A,B).
\]

只编译 carrier basis；learned channel mixing 使用系数

\[
W\in\mathbb R^{C_{\rm out}\times C_{\rm in}\times n_{\rm basis}}.
\]

D6 golden dimensions：AA/AB/BA/BB = **2/4/4/30**。

## 5. 非线性协变

SO(3) 基准通道：

\[
s_\ell(x,Z)=Z_\ell:x^{\otimes\ell},
\]

\[
q_{\ell-1}(x;Z_\ell)=\nabla_xs_\ell
=\ell Z_\ell\mathbin{\lrcorner}x^{\otimes(\ell-1)}\in A.
\]

$q$ 合法但不构成完整 finite-$G$ basis。主干必须编译

\[
\operatorname{Hom}_G
\left(\operatorname{Sym}^pA\otimes\operatorname{Sym}^qB,T\right),
\qquad T\in\{A,B\}.
\]

只允许 compiled covariants、invariant scalar gates 和 typed residuals；禁止把 A/B concat 后交给普通 MLP，也禁止逐坐标 activation、normalization、bias 或 dropout。

## 6. 多分子执行顺序

1. 建立有向邻接边。
2. 计算 $X_{ij},Q_{ij}$。
3. 构造配对 seed $(A_{ij}=X_{ij},B_{ij}=Z(Q_{ij}))$。
4. 先在同一 edge 内执行 A/B maps，保持位置—姿态配对。
5. 按 receiver 分别求和 A messages 与 B messages。
6. 对 pooled A/B 执行 finite-$G$ node mixing。
7. 输出 body-frame force/torque，再旋转到 world。

\[
H_i^A=\sum_j u_{ij}^A,\qquad H_i^B=\sum_j u_{ij}^B.
\]

低成本 cross-neighbor 项：

\[
C\!\left(\sum_j u_{ij},\sum_kv_{ik}\right)-\sum_jC(u_{ij},v_{ij})
=\sum_{j\ne k}C(u_{ij},v_{ik}).
\]

多层 message passing 时，learned sender state 必须先运输到 receiver frame：

\[
\widetilde a_{j\to i}=Q_{ij}a_j,\qquad
\widetilde b_{j\to i}^{(\ell)}=D_\ell(Q_{ij})b_j^{(\ell)}.
\]

## 7. 最小神经网络

**edge geometry → paired A/B encoder → typed sum/cross terms → 1–3 A/B blocks → independent force-A and torque-A heads → body-to-world rotation**

radial/species MLP 只能生成 invariant scalars 或 gates。运行时禁止 SVD、nullspace、群遍历、群平均和 dense $3^\ell$ tensor 构造。

\[
F_i=R_if_i,\qquad M_i=R_im_i,
\]

\[
F'_{\pi(i)}=hF_i,\qquad M'_{\pi(i)}=hM_i.
\]

torque 关于分子中心。对 $G\subset SO(3)$，force 与 torque 都使用 A；若支持 improper operations，必须区分 polar/axial carriers。

对称性不自动保证 conservative、$\sum_iF_i=0$ 或总角动量平衡。

## 8. 编译顺序与测试门

**canonical STF basis → generator representations → primitive anchors → B layout → AA/AB/BA/BB bases → polynomial covariants/couplings → versioned artifact**

最低测试：

- STF trace-free、orthonormal、跨 dtype/device 一致；
- $D(RS)=D(R)D(S)$；
- $D_\ell(g)K_\ell=K_\ell$；
- compiled maps 满足约束、维数正确；
- $Z(g_i^\top Qg_j)=D(g_i^\top)Z(Q)$；
- $N=2$ 恢复 pair A/B；
- 独立 $G^N$、全局 $SE(3)$、分子置换；
- cross-neighbor 线性实现等于显式双重求和。
