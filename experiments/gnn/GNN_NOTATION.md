## Generic Typed Stage

对 stage \(v\) 和 target type \(\tau\)，定义

\[
\boxed{
\mathcal U_{v,\tau}
\left(\mathcal Z;\mathcal K_{v,\tau}\right)
}
\]

为一个 typed stage。分号两侧含义不同：

- \(\mathcal Z\)：该 stage 可见的 named typed source pool；
- \(\mathcal K_{v,\tau}\subseteq\mathcal Z\)：额外声明的 direct same-TypeKey carriers，其中每个元素都必须属于 target type \(\tau\)。

carrier 既可在 source pool 中参与合法 covariant paths 与 invariant 构造，也作为 direct channel 进入最后的 concat-project；这不表示复制两份 runtime input。

对 stage \(v\) 中 target 为 \(\tau\) 的 registered path

\[
\kappa\in\mathcal P_{v,\tau},
\]

定义

\[
u_{ij,\ell,v,\kappa}^{\tau}
=
\sum_{\alpha}
\gamma_{ij,\ell,v,\kappa,\alpha}^{\tau}
\!\left(\mathbf s_{ij,\ell,v}\right)
C_{\tau\leftarrow\Sigma_\kappa,\alpha}
\Lambda_{\Sigma_\kappa}
\!\left(z_{ij,\ell,v}^{\kappa}\right).
\]

完整 stage 输出为

\[
\boxed{
\mathcal U_{v,\tau}
\left(\mathcal Z;\mathcal K_{v,\tau}\right)
=
(P_{v,\tau}\otimes I_{V_\tau})
\operatorname{Cat}_{c}
\left[
\mathcal K_{v,\tau},
\left\{
u_{ij,\ell,v,\kappa}^{\tau}
\right\}_{\kappa\in\mathcal P_{v,\tau}}
\right].
}
\]

\(z_{ij,\ell,v}^{\kappa}\) 不是 hidden state，而是按照 signature \(\Sigma_\kappa\) 从 \(\mathcal Z\) 中选出的 ordered typed tuple。

不同 TypeKey 只能通过 registered

\[
C_{\tau\leftarrow\Sigma_\kappa,\alpha}
\]

转换，永远不能直接 concat。

\(\mathbf s_{ij,\ell,v}\) 由 \(\mathcal Z\) 的 invariant contractions 以及可选的 fixed molecular scalars \(\mu_i,\mu_j\) 构成。Scalar 只进入 Gate，不进入 \(\Lambda_\Sigma\)、carrier 或 hidden state，也不接受 A/B 更新。

## Initial Dual-Output Message Block

### Source Pools

对有向边 \(j\to i\) 和 network layer \(\ell\)，定义初始 source pool：

\[
\boxed{
\mathcal Z_{ij,\ell}^{(0)}
=
\left\{
d_{ij},
e_{ij,\ell}^{A},
\left\{
B_i^{(r)},
h_{i,\ell}^{B^{(r)}},
B_j^{(r)},
h_{j,\ell}^{B^{(r)}}
\right\}_{r\in\mathcal R_B}
\right\}.
}
\]

Scalar 不属于 \(\mathcal Z_{ij,\ell}^{(0)}\)。Fixed molecular scalar 可以进入 \(\mathbf s_{ij,\ell,v}\)，但不产生 scalar state，也不接受 A/B 更新。对于当前完全相同的 benzene，可以直接省略 molecular scalar。

后续 stage 只把新产生的 typed values 加入 named source pool：

\[
\mathcal Z_{ij,\ell}^{(1)}
=
\mathcal Z_{ij,\ell}^{(0)}
\cup
\left\{
a_{ij,\ell,1}^{A}
\right\},
\]

\[
\mathcal Z_{ij,\ell}^{(2)}
=
\mathcal Z_{ij,\ell}^{(1)}
\cup
\left\{
b_{ij,\ell,1}^{B^{(q)}}:
q\in\mathcal R_B
\right\},
\]

\[
\mathcal Z_{ij,\ell}^{(3)}
=
\mathcal Z_{ij,\ell}^{(2)}
\cup
\left\{
b_{ij,\ell,2}^{B^{(q)}}:
q\in\mathcal R_B
\right\}.
\]

这里的 \(\cup\) 只表示向 named source pool 中加入 tensor，不表示对 tensors 执行集合运算。

### Update Schedule

第一步，生成中间 A-type edge state：

\[
a_{ij,\ell,1}^{A}
=
\mathcal U_{a1,A}
\left(
\mathcal Z_{ij,\ell}^{(0)};
d_{ij},
e_{ij,\ell}^{A}
\right).
\]

第二步，对每个 \(B^{(r)}\) 同步生成较宽的 B-type edge message：

\[
b_{ij,\ell,1}^{B^{(r)}}
=
\mathcal U_{b1,B^{(r)}}
\left(
\mathcal Z_{ij,\ell}^{(1)};
B_j^{(r)},
h_{j,\ell}^{B^{(r)}}
\right),
\qquad
r\in\mathcal R_B.
\]

第三步，对每个 \(B^{(r)}\) 同步压缩并 refine B-type edge message：

\[
b_{ij,\ell,2}^{B^{(r)}}
=
\mathcal U_{b2,B^{(r)}}
\left(
\mathcal Z_{ij,\ell}^{(2)};
B_j^{(r)},
h_{j,\ell}^{B^{(r)}},
b_{ij,\ell,1}^{B^{(r)}}
\right),
\qquad
r\in\mathcal R_B.
\]

第四步，使用全部 refined B information 更新 A-type edge state：

\[
e_{ij,\ell+1}^{A}
=
\mathcal U_{aout,A}
\left(
\mathcal Z_{ij,\ell}^{(3)};
d_{ij},
e_{ij,\ell}^{A},
a_{ij,\ell,1}^{A}
\right).
\]

\(b1\) 和 \(b2\) 分别对所有 \(r\in\mathcal R_B\) 从同一个 frozen snapshot 同步计算，不能使结果依赖 B TypeKey 的遍历顺序。

对于 target \(B^{(r)}\)，其他 \(B^{(q)}\) 可以通过合法 registered path 参与计算，但只有 same-TypeKey \(B^{(r)}\) 可以作为 direct carrier。

该 block 返回

\[
\boxed{
\left(
e_{ij,\ell+1}^{A},
\left\{
b_{ij,\ell,2}^{B^{(r)}}
\right\}_{r\in\mathcal R_B}
\right).
}
\]

其中：

- \(e_{ij,\ell+1}^{A}\) 是保留在 edge 上的 A-type state；
- \(b_{ij,\ell,2}^{B^{(r)}}\) 是有向边 \(j\to i\) 上尚未聚合的 \(B^{(r)}\)-type message；
- aggregation 和 node update 不属于当前模块。

Receiver features

\[
B_i^{(r)},
\qquad
h_{i,\ell}^{B^{(r)}}
\]

可以参与 registered covariant paths 和 Gate invariants，但不能作为每条 B-message 的 direct carrier。否则未来执行 \(\sum_j\) 时，会产生 degree-scaled receiver state。

## Symbol Definitions

| 符号 | 含义 |
|---|---|
| \(i,j\) | Molecular-node indices |
| \(j\to i\) | 从 sender \(j\) 指向 receiver \(i\) 的有向边 |
| \(\ell\) | Network-layer index |
| \(v\) | Block-stage label；\(v\in\{a1,b1,b2,aout\}\) |
| \(G\) | Construction 时由 generators 给定的 registered symmetry group |
| \(A\) | Registered A representation type |
| \(B^{(r)}\) | Label 为 \(r\) 的 registered B representation type |
| \(\tau\) | 任意 target representation type |
| \(\mathcal R_B\) | 当前 `TypeCatalog` 中所有 B TypeKey labels 的集合 |
| \(r\) | 当前 target B TypeKey label |
| \(q\) | 遍历 source B bank 的 B TypeKey label |
| \(V_\tau\) | Type \(\tau\) 的 representation space |
| \(I_{V_\tau}\) | \(V_\tau\) 上的 identity map |
| \(X_i\) | Node \(i\) 在 single-root frame 中的 molecular center |
| \(d_{ij}=X_j-X_i\) | Fixed raw A-type edge displacement；不是 scalar distance |
| \(e_{ij,\ell}^{A}\) | Layer \(\ell\) 的 hidden A-type edge state |
| \(B_i^{(r)}\) | Node \(i\) 的 raw pose descriptor of type \(B^{(r)}\) |
| \(h_{i,\ell}^{B^{(r)}}\) | Node \(i\)、layer \(\ell\) 的 hidden \(B^{(r)}\) state |
| \(a_{ij,\ell,1}^{A}\) | Block 内第一个 A refinement；临时 edge state |
| \(b_{ij,\ell,1}^{B^{(r)}}\) | Block 内第一个、较宽的 \(B^{(r)}\) edge message |
| \(b_{ij,\ell,2}^{B^{(r)}}\) | Block 内压缩后返回的 \(B^{(r)}\) edge message |
| \(\mathcal Z_{ij,\ell}^{(n)}\) | 第 \(n\) 个 stage boundary 的 named typed source pool |
| \(\mathcal K_{v,\tau}\) | Stage \(v\)、target \(\tau\) 的 same-TypeKey direct carriers |
| \(\mathcal U_{v,\tau}\) | 由合法 typed paths、Gate 和 carriers 组成的 target-\(\tau\) stage |
| \(\mathcal P_{v,\tau}\) | Stage \(v\) 中 target 为 \(\tau\) 的 registered path set |
| \(\kappa\) | 一条 registered path，包括 signature 和 source-role assignment |
| \(\Sigma_\kappa\) | Path \(\kappa\) 的 ordered typed source signature |
| \(z_{ij,\ell,v}^{\kappa}\) | 按 \(\Sigma_\kappa\) 从 source pool 取出的 runtime ordered tuple |
| \(\Lambda_{\Sigma_\kappa}\) | Signature \(\Sigma_\kappa\) 的 normalized tensor/polynomial source lift |
| \(W_{\Sigma_\kappa}\) | \(\Lambda_{\Sigma_\kappa}\) 所在的 lifted source representation space |
| \(\alpha\) | \(\operatorname{Hom}_G(W_{\Sigma_\kappa},V_\tau)\) 的 intertwiner-basis index |
| \(C_{\tau\leftarrow\Sigma_\kappa,\alpha}\) | Compiler 产生的 frozen equivariant map |
| \(\mathbf s_{ij,\ell,v}\) | Stage-local invariant Gate descriptor；不是 scalar hidden state |
| \(\gamma_{ij,\ell,v,\kappa,\alpha}^{\tau}\) | Gate 产生的 signed scalar coefficient |
| \(u_{ij,\ell,v,\kappa}^{\tau}\) | Path \(\kappa\) 生成的 target-\(\tau\) covariant branch |
| \(\operatorname{Cat}_{c}\) | 只沿 typed multiplicity/channel axis 进行拼接 |
| \(P_{v,\tau}\) | Trainable channel projection；不作用于 representation coordinates |
| \(\otimes\) | Tensor product |
| \(P_{v,\tau}\otimes I_{V_\tau}\) | 只混合 channels，同时保持 representation action |
| \(c\) | Typed multiplicity-channel count；不是 tensor rank |
| \(\mu_i\) | Optional fixed invariant molecular scalar data；只能进入 Gate descriptor |

上标 \(B^{(r)}\) 始终表示 representation type，不是 tensor power。Block 内 refinement 编号 \(1,2\) 因此放在下标中。

C17 的初始 channel prior 是

\[
B(c=2)\longrightarrow B(c=1),
\]

这里的 \(2\to1\) 是 channel contraction，不是 \(B^{(r)}\) rank transformation。