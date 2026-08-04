# A/B Finite-Group Mathematical Module Contract

本文只规定可复用、可独立测试的数学模块。不要在这里加入 message passing、attention、邻居构图、网络深度、训练目标或完整 force-field 架构。

状态约定：

- **FROZEN**：接口、轴顺序和变换律已经冻结；修改必须提升 convention version 并更新性质测试。
- **DESIGN**：仅冻结数学原则；不得假定已有实现或自行虚构公共 API。
- Python 公共参数统一使用 `rank`；数学公式使用 \(\ell\)。

## 1. 单一公共 frame 与群作用

令 \(H\le SO(3)\) 为同种分子的 proper rotational symmetry group。它不包含反射或反演；benzene 的 \(H\) 是纯旋转 \(D_6\)，不是通常含 improper operations 的 \(D_{6h}\)。

世界坐标变换写成

\[
x'_{\pi(i)}=Sx_i+t,\qquad
R'_{\pi(i)}=SR_i h_i,
\qquad S\in SO(3),\ h_i\in H.
\]

物理参考分子记为 \(0\)。整个前向过程只建立一次公共 frame：

\[
X_i=R_0^\top(x_i-x_0),\qquad
Q_i=R_0^\top R_i.
\]

于是

\[
X'_{\pi(i)}=h_0^\top X_i,\qquad
Q'_{\pi(i)}=h_0^\top Q_i h_i.
\]

所有非参考分子共享同一个左侧 gauge \(h_0^\top\)；每个 sender 的右侧 gauge 是 \(h_i\)。边位移只在这个公共 frame 中构造：

\[
\Delta X_{ij}=X_j-X_i,\qquad
\Delta X'_{\pi(i)\pi(j)}=h_0^\top\Delta X_{ij}.
\]

不得为每条边或每个 receiver 再建立 local frame。参考分子是带标记的根；普通 permutation 只重排非根槽位。re-rooting 是独立问题，不属于当前模块合同。

局部 A 型力满足 \(f'_{\pi(i)}=h_0^\top f_i\)，统一恢复到世界坐标：

\[
F_i^{\rm world}=R_0f_i.
\]

## 2. FROZEN：现有数学模块的短接口

详细参数、异常和 dtype/device 行为以各文件 module docstring 为准；本文件只保留跨模块合同。

\[
V_\ell=\operatorname{Sym}_0^\ell(\mathbb R^3),\qquad
d_\ell=2\ell+1,
\qquad
D_\ell(RS)=D_\ell(R)D_\ell(S).
\]

| 模块 | 冻结接口摘要 |
|---|---|
| `_compiler_utils.py` | 内部离线 CPU-float64 canonical nullspace/range/projector 工具；不是 runtime API |
| `stf_space.py` | STF/symmetric/dense 转换、`stf_basis`、`stf_power_components`、固定尺度/相位的 `stf_tensor_coupling` |
| `stf_rep.py` | `stf_representation(R, rank)`：\((\ldots,3,3)\to(\ldots,d_\ell,d_\ell)\)，仅 proper rotations |
| `anchor_compiler.py` | `compile_anchors(generators, ranks) -> AnchorCompilation`：离线编译 fixed/generated/primitive bases 与诊断 |
| `pose_encoding.py` | `PoseEncoder`、`encode_blocks`、`encode/forward`、`representation`；block 为 \((\ldots,d_\ell,m_\ell^K)\)，flatten 为 rank 升序和 anchor-major |
| `intertwiner_compiler.py` | `compile_intertwiners` 返回 \((n_{T\leftarrow U},d_T,d_U)\) Frobenius-orthonormal basis；另有 residual 与 direct-sum 接口 |
| `covariants.py` | `scalar_contraction` 输出 \((\ldots,m_\ell^K)\)；`vector_covariant` 输出 \((\ldots,m_\ell^K,3)\)，是固定 pose block 时的位置偏导 |

必须保留以下跨模块性质：

\[
\Gamma_{\ell_1\ell_2}^{L}
\bigl(D_{\ell_1}(R)a,D_{\ell_2}(R)b\bigr)
=D_L(R)\Gamma_{\ell_1\ell_2}^{L}(a,b),
\]

\[
K_\ell^{\rm fix}=\operatorname{Fix}_H(V_\ell),\qquad
K_\ell^{\rm prim}=K_\ell^{\rm fix}\ominus K_\ell^{\rm gen},
\]

\[
Z_i^{(\ell)}=D_\ell(Q_i)K_\ell^{\rm prim}
\mapsto D_\ell(h_0^\top)Z_i^{(\ell)}.
\]

standard STF coupling 是 \(SO(3)\)-equivariant 子目录；linear compiler 编译的是指定表示的 \(H\)-intertwiner。所有 master bases/compiler artifacts 离线生成，runtime 不得重新求基或 SVD。

### DESIGN：商空间分离证书

定义

\[
H_K=
\left\{q\in SO(3):D_\ell(q)K_\ell=K_\ell
\text{ for every encoded }\ell\right\}.
\]

右不变性只证明编码定义在某个商空间上。在声称它分离 \(SO(3)/H\) 前，必须由项目自己的 stabilizer/separation test 认证 \(H_K=H\)；否则实际编码的是更粗的 \(SO(3)/H_K\)。

## 3. Typed A/B/S layout

\[
A=V_1|_H,
\qquad
B=\bigoplus_{\ell\in\mathcal L_B}V_\ell|_H,
\qquad
S=V_0.
\]

benzene proper \(D_6\) 的 primitive ranks 为

\[
\mathcal L_B=\{2,6\},
\qquad B=V_2\oplus V_6.
\]

统一 feature block 采用

\[
(\ldots,C_T,d_T),
\]

其中 \(C_T\) 是 learned channel/multiplicity，\(d_T\) 是 representation coordinate。群只作用于最后一轴：

\[
x_T\mapsto(I_{C_T}\otimes\rho_T(h))x_T.
\]

必须区分 pose anchor multiplicity \(m_\ell^K\) 与 learned channel count \(C_{B_\ell}\)。不得将 representation 轴无类型拼接后交给普通 MLP，也不得逐表示分量施加普通激活。

## 4. DESIGN：A/A、A/B、B/A、B/B linear channels

四类 channel 是 linear Hom spaces：

命名中第一个类型是 source，第二个类型是 target；正式实现名称应优先使用 `A_to_A`、`A_to_B`、`B_to_A`、`B_to_B`，避免方向歧义。

| Channel | 数学空间 |
|---|---|
| A/A | \(\operatorname{Hom}_H(A,A)\) |
| A/B | \(\operatorname{Hom}_H(A,B)\) |
| B/A | \(\operatorname{Hom}_H(B,A)\) |
| B/B | \(\operatorname{Hom}_H(B,B)\) |

若 \(\{M_\alpha^{T\leftarrow U}\}_{\alpha=1}^{n_{T\leftarrow U}}\) 是冻结 carrier basis，则带 learned channels 的一般线性作用为

\[
y_{r,a}
=\sum_{s,\alpha,b}
W_{rs\alpha}
(M_\alpha^{T\leftarrow U})_{ab}
x_{s,b}.
\]

实现必须具有以下性质：

- \(M_\alpha\) 是离线资产；\(W\) 只作用于 channel 和 intertwiner multiplicity。
- 在线仅做 batched contraction，保留 dtype/device 和 autograd。
- \(n_{T\leftarrow U}=0\) 时该 path 不存在，不得伪造零 basis 后宣称存在。
- 非标量 bias 必须属于 \(\operatorname{Fix}_H(T)\)；默认应禁用。
- A/A、A/B、B/A、B/B 是 carrier 类型，不等同于网络层或消息传递方向。

对 proper \(D_6\)（order 12）、\(A=V_1\)、\(B=V_2\oplus V_6\)，且 \(V_2,V_6\) 各只有一个 carrier copy 时，character 公式

\[
\dim\operatorname{Hom}_H(U,T)
=\frac1{|H|}\sum_{h\in H}
\overline{\chi_T(h)}\chi_U(h)
\]

给出 golden basis dimensions：

\[
\boxed{
n_{AA},n_{AB},n_{BA},n_{BB}=2,4,4,30
}.
\]

这些数字是 carrier-basis 数量的回归合同，不是 learned feature-channel 数量。

## 5. DESIGN：C channels

`C` 不是一个手写 tensor，也不是普通 MLP；它是按 source signature 和 target type 索引的 multilinear intertwiner catalog。

### 5.1 Bilinear C

对已注册表示 \(U,V,T\)，定义

\[
\mathcal C_{T;U,V}
=\operatorname{Hom}_H(U\otimes V,T).
\]

\(C\in\mathcal C_{T;U,V}\) 当且仅当

\[
C(\rho_U(h)u,\rho_V(h)v)
=\rho_T(h)C(u,v),
\qquad h\in H.
\]

固定 tensor-product index

\[
\operatorname{pair}(i,j)=i\,d_V+j,
\qquad
\rho_{U\otimes V}(h)=\rho_U(h)\otimes\rho_V(h).
\]

选择 Frobenius-orthonormal basis

\[
C_\alpha^{T;U,V}
\in\mathbb R^{d_T\times d_U\times d_V},
\qquad
\alpha=1,\ldots,\mu_{T;U,V}.
\]

冻结 basis layout：

\[
(\mu_{T;U,V},d_T,d_U,d_V).
\]

若将 \(C_\alpha\) reshape 为 \(M_\alpha\in\mathbb R^{d_T\times d_Ud_V}\)，则

\[
M_\alpha
(\rho_U(h)\otimes\rho_V(h))
=\rho_T(h)M_\alpha.
\]

完整 multiplicity 为

\[
\mu_{T;U,V}
=\frac1{|H|}\sum_{h\in H}
\overline{\chi_T(h)}\chi_U(h)\chi_V(h).
\]

generator-nullspace dimension、完整群 character multiplicity 和 Reynolds projection rank 必须一致。

### 5.2 Runtime contraction

对

\[
x^U\in\mathbb R^{C_U\times d_U},
\qquad
x^V\in\mathbb R^{C_V\times d_V},
\]

固定几何 primitive 为

\[
z_{pq\alpha a}
=\sum_{ij}
(C_\alpha)_{aij}x^U_{p,i}x^V_{q,j}.
\]

一般输出只能通过 invariant-controlled channel coefficients 组合：

\[
y_{r,a}
=\sum_{p,q,\alpha}
\theta_{rpq\alpha}(I)
z_{pq\alpha a},
\]

其中 \(I\) 是严格 \(H\)-invariant scalars，\(\theta\) 是光滑标量函数。learned coefficients 不得自由作用于 representation index \(a\)。

### 5.3 Target families

\[
C_A^{U,V}
=\operatorname{Hom}_H(U\otimes V,A),
\]

\[
C_{B_L}^{U,V}
=\operatorname{Hom}_H(U\otimes V,V_L),
\qquad L\in\mathcal L_B,
\]

\[
C_S^{U,V}
=\operatorname{Hom}_H(U\otimes V,V_0).
\]

因此 `C_A`、`C_B` 是 catalog views；`C_B` 必须保留 target rank \(L\)，不能压成一个无类型 tensor。

完整 \(SO(3)\) Clebsch–Gordan channels 是合法但通常更小的子空间：

\[
\operatorname{Hom}_{SO(3)}(U\otimes V,T)
\subseteq
\operatorname{Hom}_{H}(U\otimes V,T).
\]

不得用 \(SO(3)\) triangle rule 判定 finite-\(H\) path 一定不存在。

### 5.4 Proper \(D_6\) bilinear golden dimensions

对完整 finite-\(H\) catalog、\(H=D_6\)、\(A=V_1\)、\(B=V_2\oplus V_6\)，且不计 learned/anchor channel multiplicity，ordered bilinear Hom spaces 的维数为：

| Source signature | target A | target B |
|---|---:|---:|
| \(A\otimes A\) | 3 | 16 |
| \(A\otimes B\) | 16 | 80 |
| \(B\otimes A\) | 16 | 80 |
| \(B\otimes B\) | 80 | 492 |

这些数字来自

\[
\dim\operatorname{Hom}_H(U\otimes V,T)
=\frac1{|H|}\sum_{h\in H}
\overline{\chi_T(h)}\chi_U(h)\chi_V(h),
\]

是完整 ordered carrier catalog 的 golden tests。若使用 exchange-symmetric、exchange-antisymmetric、\(SO(3)\)-only 或截断 catalog，维数会改变，必须在 `CatalogSpec` 中标记为 restricted，不能继续声称是完整 finite-\(H\) basis。

### 5.5 Higher-order C

一般 \(p\)-linear catalog 为

\[
\operatorname{Hom}_H
\left(U_1\otimes\cdots\otimes U_p,T\right).
\]

同一变量重复出现时使用

\[
\operatorname{Hom}_H
\left(
\operatorname{Sym}^{p_1}U_1
\otimes\cdots\otimes
\operatorname{Sym}^{p_k}U_k,
T
\right).
\]

“完整”仅表示：对已声明的

\[
\text{CatalogSpec}
=(H,\ell_{\max},p_{\max},\text{allowed signatures})
\]

中的每个 signature，返回完整 Hom-space basis。不得声称有限 catalog 覆盖所有次数的 invariant ring。

### 5.6 Differentiability and assets

固定 bilinear primitive 是多项式：

\[
D_uC(u,v)[\delta u]=C(\delta u,v),
\qquad
D_vC(u,v)[\delta v]=C(u,\delta v).
\]

offline SVD/nullspace 不参与训练；runtime 只能包含固定 basis contractions、求和及光滑 invariant gating。每个 C artifact 必须保存 group/convention fingerprint、signature、basis layout、dimension、character prediction、singular threshold/gap、equivariance residual 和 basis version。basis 非唯一，因此 checkpoint 不得依赖重新 SVD 获得相同坐标。

## 6. 模块验收，不是网络设计

新增模块只需通过以下性质测试：

- STF round-trip、trace、orthonormality、\(D_\ell\) homomorphism。
- standard coupling 的 \(SO(3)\) equivariance、phase/scale golden tests。
- anchor fixed/generated/primitive 正交分解与 pose 左协变/右不变。
- 商空间模块验证 \(H_K=H\)，或明确拒绝“separating encoding”声明。
- linear Hom basis 的 character dimension、orthonormality 和 generator/full-group residual。
- \(D_6\) 的 2/4/4/30 golden dimensions。
- 完整 ordered bilinear \(D_6\) catalog 的 3/16、16/80、16/80、80/492 golden dimensions。
- C 的 tensor flatten golden test、nullity/character/Reynolds dimension 一致、forward equivariance、两个输入分别 `gradcheck`。
- runtime 不得包含 `.item()` 数据分支、SVD、nullspace、CPU round-trip 或 basis 重编译。
- convention/group/signature 不匹配的资产必须拒绝加载。

不要在这些测试中搭建完整网络；只使用最小 typed tensors 验证数学模块。

## 7. References

下列文献只支持对应的一般表示论或 invariant-theory 原理；项目特定的 STF normalization、flatten convention、finite catalog 截断以及 \(H_K=H\) 必须由本项目自己的合同与测试证明。

- Risi Kondor, Zhen Lin, Shubhendu Trivedi — [Clebsch–Gordan Nets: a Fully Fourier Space Spherical Convolutional Neural Network](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a3fc981af450752046be179185ebc8b5-Abstract.html).
- Leon Lang, Maurice Weiler — [A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels](https://openreview.net/forum?id=ajOrOhQOsYx).
- Marc Finzi, Max Welling, Andrew Gordon Wilson — [A Practical Method for Constructing Equivariant Multilayer Perceptrons for Arbitrary Matrix Groups](https://proceedings.mlr.press/v139/finzi21a.html).
- Taco S. Cohen, Mario Geiger, Maurice Weiler — [A General Theory of Equivariant CNNs on Homogeneous Spaces](https://proceedings.neurips.cc/paper/2019/hash/b9cfe8b6042cf759dc4c0cccb27a6737-Abstract.html).
- Risi Kondor, Shubhendu Trivedi — [On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups](https://proceedings.mlr.press/v80/kondor18a.html).
- Dmitry Yarotsky — [Universal Approximations of Invariant Maps by Neural Networks](https://link.springer.com/article/10.1007/s00365-021-09546-1).
- Gerald W. Schwarz — [Smooth Functions Invariant Under the Action of a Compact Lie Group](https://www.sciencedirect.com/science/article/pii/0040938375900361).
