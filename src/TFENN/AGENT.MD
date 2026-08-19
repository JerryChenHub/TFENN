# TFENN Agent Key Information

本文只保留实现与实验设计中不可混淆的项目约定。完整数学记号以
`MATH_NOTATION.MD`、`MATH_NOTATION_2.md` 和
`AB_Covariant_Network_Design_Core.tex` 为准。网络拓扑仍在探索中；不要从旧实验结构反推新的固定架构。

## 1. 项目范围与群

- 当前以 benzene pair 为开发与验证对象，长期目标是适用于编译器支持的任意 proper rotation group $G\leq SO(3)$，而不是构造 benzene 专用网络。
- 项目 shorthand `D6~` 指由一个主轴 $60^\circ$ 旋转和一个面内轴 $180^\circ$ 旋转生成的 proper rotation group，正式数学记号为 $D_6^{\mathrm{rot}}$；它不含镜像或反演。
- 群表示、pose anchors 和 intertwiner bases $C$ 必须由输入 generators 离线编译；group-independent symmetric lifts $U_{d,p}$ 由 $(d,p)$ 离线构造。不得在模型或编译器中硬编码或预存 benzene 的群元素、character/irrep 表、path 维数或 $C$ basis。
- TFENN 主模型不使用 group convolution 或 runtime group/Reynolds averaging。明确标注的外部 MLP baseline 是唯一例外，且不得称为 TFENN covariant layer。

## 2. 唯一 root frame 与原生入口

世界坐标中，分子 $i$ 的中心和姿态为

\[
x_i\in\mathbb R^3,\qquad O_i\in SO(3).
\]

选定 root molecule $0$ 后只做一次坐标变换：

\[
X_i=O_0^\top(x_i-x_0),\qquad R_i=O_0^\top O_i.
\]

代码和实验表中的简写必须按下列含义解释：

| 简写 | 含义 |
|---|---|
| raw `x` | 当前 pair/node 的 root-frame displacement $X_i$，不是 world $x_i$ |
| raw `R` | relative orientation matrix $R_i$ |
| raw `r` | $r=\operatorname{PoseEncoder}(R)$ 产生的完整 typed B bank；不是旋转矩阵、STF rank 或单个 B block |

原生 covariant ingress 固定为

\[
x\longrightarrow A,\qquad
R\longrightarrow \operatorname{PoseEncoder}\longrightarrow
r=\{B^{(s)}\}_{s\in\mathcal R_B}.
\]

因此 raw pose 不应先进入 A 再转换到 B。所有 A/B hidden tensors 位于同一 root fiber；声明过的同型特征可以直接 mixing，不需要 transporter。最终 local A-type force 用 $O_0$ 恢复到 world frame。

## 3. Type、stage、channel 与 width

- $A$ 是物理 vector type；$B^{(r)}$ 是 pose-derived typed block，右上标 $r$ 只是 B-block/type label。
- `a1`, `a2`, `b1`, `out` 是计算图中的 stage/node 名，不表示 channel 数或表示阶数。
- stage 记为 $v$，其 execution level 记为 $\ell(v)$；同一 level 的并行 A/B stages 仍是不同节点。
- `a2:A[16]` 表示 stage `a2` 含 16 个 A feature channels。
- `B[c]` 表示每个已注册 B TypeKey 有 $c$ 个 learned channels；若不同 block 宽度不同，必须写成 $B^{(r)}[c_r]$。channel 轴与 representation coordinate 轴不可混淆。
- $w_v^{\mathrm{gate}}$ 或配置中的 `W` 是 stage $v$ 的 Gate MLP hidden width；只有同一 level 的 stages 共享宽度时才写 $w_\ell^{\mathrm{gate}}$。它不是 A/B channel 数、STF rank 或 mixing rank。

### Latent mixing rank

dense channel mixing 会直接学习 output-channel 与所有 input-channel 组合的系数。factorized mixing 则先将每个 input slot 投影到共享 latent channel

\[
q=1,\ldots,r_{\mathrm{mix}},
\]

在每个 $q$ 上计算 gated covariant paths，再用 $P_{\mathrm{out}}$ 投影到 output channels。$r_{\mathrm{mix}}$ 就是这些中间 channel mixtures 的数量；它控制 channel mixing 的低秩容量与参数量。

$r_{\mathrm{mix}}$ 不是 $R_i$、STF rank、B-block rank、stage 数、实际 feature-channel 数或 $w_v^{\mathrm{gate}}$。模型必须声明使用 dense、factorized 还是 diagonal mixing；factorized 模式必须给出 $r_{\mathrm{mix}}$。

## 4. 统一 activation 与参数生命周期

所有 typed paths 使用同一数学接口，但按 source signature $\Sigma$ 与 target type $\tau$ 取得各自的 frozen basis 和 coefficient head：

\[
\operatorname{Act}_{v,\kappa}(z_1,\ldots,z_k;\mathbf s)
=\sum_\alpha
\gamma_{v,\kappa,\alpha}(\mathbf s)
C_{\tau\leftarrow\Sigma,\alpha}
\Lambda_\Sigma(z_1,\ldots,z_k).
\]

Here \(\kappa=(\tau\leftarrow\Sigma,\text{source roles})\) identifies the
registered path. With channels, \(\gamma_{v,\kappa}\) is the whole coefficient
tensor; \(\gamma_{v,\kappa,\alpha}\) suppresses its channel indices.

- **Offline/frozen**：generator-derived representation/anchor data、group-independent normalized symmetric-power embedding $U$、intertwiner basis $C$ 及其 metadata。
- **Trainable**：Gate trunk/head 参数，以及 dense、factorized 或 diagonal channel-mixing parameters。
- **Runtime values**：raw/hidden covariants、invariants $\mathbf s$、polynomial lift $\Lambda_\Sigma$ 和 Gate 输出 $\gamma(\mathbf s)$。$\gamma$ 本身不是存储参数。

Gate trunk 可以非线性。当前 coefficient head 的默认输出是 signed, unbounded identity，即最后一个线性层后不再加 sigmoid/tanh/SiLU；这不会使整个网络变成线性网络。

## 5. Covariant flow policy：每个模型必须声明

Strict flow 不是强制架构。每个模型必须明确选择并列出实际 covariant edges：

| 名称 | 定义 |
|---|---|
| `STRICT_DECLARED_FLOW` | 每个 stage 只读取拓扑为它指定的 parent outputs；不自动携带 earlier-history bank |
| `EXPLICIT_REUSE_FLOW` | 允许 selected earlier-layer outputs 直接进入后续 stage，但每一条 reuse edge 必须显式列出 |
| `DENSE_HISTORY_FLOW` | 后续 stage 读取所有符合声明规则的 earlier outputs；它是 fully enumerated reuse graph 的简写 |

Raw covariant visibility 另行声明：

| 名称 | 定义 |
|---|---|
| `STEM_ONLY_RAW` | raw $x\to A$ 与 $R\to r\to B$ 只在原生入口出现 |
| `EXPLICIT_RAW_REREAD` | 后续指定 stages 可以再次读取 raw typed A/B；必须逐 stage 列出这些 edges |
| `EVERY_STAGE_RAW` | 每个 compatible stage 都重新读取明确允许的 raw typed sources |

同型 bypass 与上述 flow policy 不冲突：任何**已经声明**的同一 TypeKey edge 都可以通过 carrier identity 保留原协变量，再做 channel projection/mixing。它是同型协变 flow，不是自动读取全部历史，也不是额外的 additive ResNet identity branch。跨类型 edge 必须使用 registered intertwiners $C$。

## 6. Invariant context 是独立策略

Covariant flow 决定哪些 A/B tensors 进入 covariant paths；invariant-context policy 决定 Gate 可以读取哪些 scalar invariants。二者必须分开声明。

- `RAW_ONLY_INVARIANTS`：只使用 raw geometric/pose invariants。
- `RAW_PLUS_PARENTS_INVARIANTS`：raw invariants 加当前 stage 已声明 parents 产生的 invariants。
- `DENSE_HISTORY_INVARIANTS`：使用 raw invariants 与全部 earlier hidden outputs 产生的 invariants。
- `EXPLICIT_INVARIANTS`：显式列出 invariant sources 与 scalar signatures。

即使采用 `STRICT_DECLARED_FLOW` 与 `STEM_ONLY_RAW`，每一层 Gate 仍可重复使用允许的 raw invariants；复用 invariant scalars 不会增加 covariant bypass。必须从 radial/raw、unary、$\operatorname{Sym}^p$、raw--hidden、hidden--hidden 与 STF shortcuts 中逐类列出实际启用的 invariant families 及 source roles，不得只写 `FULL`。

## 7. 每个模型设计必须报告的内容

仅写“使用 A/B 交错”或“使用 Invariant Gate”不够。每个模型至少要明确：

1. typed stage DAG、每条 covariant edge，以及每个 stage 的 execution level 和 `A[c]` / `B[c]` channels；
2. covariant-flow、raw-covariant-access 与 invariant-context policy；
3. 每条 $A\to B$、$B\to B$、$B\to A$ 和 $A\to A$ edge 使用的 path family 与 source signature；
4. $A\to B$ 是 linear lift、self-polynomial $\operatorname{Sym}^p A\to B$，还是 mixed multi-slot path；
5. $B\to B$ 是 linear same/cross-block mixing、independent $B^{(s)}\otimes B^{(t)}\to B^{(r)}$，还是 self-polynomial $\operatorname{Sym}^p B^{(s)}\to B^{(r)}$；
6. 是否保留 declared same-TypeKey bypass，以及 full 或 restricted $C$-basis catalog；
7. Gate invariant roles、trunk depth/nonlinearity、$w_v^{\mathrm{gate}}$、gamma-head parameterization 和 coefficient-output activation；
8. dense/factorized/diagonal channel mixing；若 factorized，报告 $r_{\mathrm{mix}}$。

GNN、message passing 和最终网络深度尚未冻结，本文件不替它们规定架构。
