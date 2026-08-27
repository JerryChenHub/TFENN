# E311 Message Block：12 项 GNN 实验设计（V1）

## 1. 本轮问题与冻结项

目标不是继续搜索更大的 Message Block，而是回答三个图级问题：

1. 已收敛的单层 pair block 放入图封装后是否仍然正确；
2. 多层中的收益来自普通 edge refinement，还是来自跨边的 (B) 聚合；
3. 当前状态流能否学到可辨识的真实三体依赖。

所有实验冻结同一个 E311-derived 内核：

| 项目 | 固定值 |
|---|---|
| typed schedule | (A1\rightarrow B(c=2)\rightarrow B(c=1)\rightarrow A1) |
| edge (A) channels | 1 |
| hidden (B^{(r)}) channels | 每个 TypeKey 1 |
| wide / narrow (B) channels | 2 / 1 |
| Gate | width 8, SiLU, signed unbounded readout |
| radial / inverse-power bank | 与已收敛单层模型完全相同 |
| generic raw-mixed covariants | 关闭 |
| force readout | 只读取最后一层，不累加中间层 force |

本轮只改变：监督口径、层数、层间 (B) 聚合、层间参数共享和训练/测试图大小。
不增加 path、不增宽 Gate、不改 invariant bank。

## 2. 多层核心状态流

初始状态为 (e_0=0, h_0=0)。第 (ell) 层调用未修改的 Message Block：

\[
Y_\ell=
\operatorname{Block}_\ell
(X,O,\mathcal E,h_\ell,e_\ell),
\qquad
e_{\ell+1}=Y_\ell.\texttt{edge\_a\_world}.
\]

层间 directed (B) message 始终在 receiver local frame 中聚合：

\[
\bar m_{i,\ell}^{B^{(r)}}=
\begin{cases}
0,&\texttt{none},\\[2pt]
\displaystyle\sum_{j\in\mathcal N(i)}m_{j\to i,\ell}^{B^{(r)}},
&\texttt{sum},\\[8pt]
\displaystyle\frac{1}{\max(1,\deg i)}
\sum_{j\in\mathcal N(i)}m_{j\to i,\ell}^{B^{(r)}},
&\texttt{mean}.
\end{cases}
\]

node update 是 Message Block 外部的同 TypeKey、无 bias、channel-only 投影：

\[
h_{i,\ell+1}^{B^{(r)}}=
(P_{\ell,r}\otimes I)
\operatorname{Cat}_c
\left[h_{i,\ell}^{B^{(r)}},
\bar m_{i,\ell}^{B^{(r)}}\right].
\]

因此它不会改变 representation type，也没有向 path/Gate 增加容量。
`none` 对照仍传播 (e_{\ell+1})，只切断跨边 (B) 通信；这是区分“深度”和“多体通信”的关键。

最终预测只取最后一层：

\[
F_i=Y_{L-1}.\texttt{node\_force\_world}_i,
\qquad
\sum_iF_i=0.
\]

## 3. 十二项实验

| ID | 配置 | 唯一目的 | 决定性比较 | 预期 / 失败解释 |
|---|---|---|---|---|
| X01 | 双 benzene；1 层；pair-force 监督 | 复现已经收敛的单层基线，锁定数据、尺度与 checkpoint 口径 | 当前收敛记录 | 超出 seed 波动表示新 runner 或数据口径已改变，不能继续解释 GNN 差异 |
| X02 | 冻结 X01；(N=3,4,5)；逐 pair 显式求和 | 在图训练前测量 pair 误差如何累积，并验收 wrapper | 显式 pair loop vs 单层 graph wrapper | 两条路径应数值一致；不一致是 frame、edge orientation 或 signed scatter 错误 |
| X03 | 五 benzene；1 层；只监督 node force | 检查 full-graph scatter 与总力监督的 credit assignment | X04；X02 仅作 transfer 诊断 | 若明显差于 X04，瓶颈是监督分解，不应先扩 path/Gate |
| X04 | 五 benzene；1 层；监督 OPLS pair force | 建立相同 edge 表示的 pair-supervised optimization reference | X03 | X03–X04 gap 定量给出 credit-assignment 代价，不把 X04 称为理论上限 |
| X05 | 五 benzene；2 层独立；`none` | 隔离额外 edge-local refinement / 参数量的作用 | X06（同深度、同参数量） | 相对 X03 的提升只能解释为更深 pair refinement，不能作为多体证据 |
| X06 | 五 benzene；2 层独立；`sum` | 在固定深度下测量跨边 (B) 通信 | X05 | OPLS 是 pair-additive 负对照，不应需要巨大收益；退化通常指向聚合尺度或 node update |
| X07 | 五 benzene；2 层 `sum`；`shared_all` | 检查 recurrent block 的参数效率 | X06 | 接近 X06 且参数更少则支持共享；必须同时报告共享 RunningRMS 的语义 |
| X08 | 五 benzene；3 层独立；`sum` | 检查第三轮通信是否有益及深层稳定性 | X06 | OPLS 应较早平台；(B) norm/gradient 增长或验证变差表示深度不稳 |
| X09 | (N=3,4) 训练、(N=5) 测试；2 层 `sum` | 检查 extensivity 与 graph-size 外推 | X10 | pair-additive 目标下每节点误差应大致稳定 |
| X10 | 与 X09 完全相同；仅改 `mean` | 判断 mean 是否抹去随 degree 增长的环境强度 | X09 | 固定 (N) 时 sum/mean 只差常数，故必须用 size shift 才有辨识力 |
| X11 | 保守三体链；2 层独立；`none` | 建立没有跨边通信时的不可约下界 | X12 | edge ((0,1)) 对 node 2 的 Jacobian 应保持零，test error 存在平台 |
| X12 | 同一三体链；2 层独立；`sum` | 正面证明聚合产生可学习的三体依赖 | X11 | Jacobian 变为非零且误差显著下降，才算 full-GNN 多体通信成立 |

这里的 “12 项” 是 12 个预先注册的实验条件，而不是事后从同一次 sweep 中挑选结果。

## 4. X11/X12 的可辨识三体正对照

不要先用含 (0\!-!2) 边的全连接 ATM 数据。全连接图会让单层 block 通过直接边读取 node 2，削弱“必须聚合”的判别力。

使用三节点链：

\[
\mathcal E=\{\{0,1\},\{1,2\}\}.
\]

构造平滑截断的保守角势：

\[
U_3=
\lambda f_c(r_{01})f_c(r_{12})
\left(
\widehat r_{10}^{\top}\widehat r_{12}-c_0
\right)^2,
\qquad
F_i=-\nabla_{x_i}U_3.
\]

选择远离 (r=0) 与 cutoff 边界的采样区间，并用 float64 autograd 生成标签。
该构造满足保守性与总力为零，而且一般有

\[
\frac{\partial F_0}{\partial x_2}\neq0.
\]

当层间聚合关闭时，edge ((0,1)) 的输入闭包不含 node 2；增加 edge-local 深度也不能改变这一点。
开启 sum 后，信息路径为

\[
2\longrightarrow m_{2\to1,0}^{B}
\longrightarrow h_{1,1}^{B}
\longrightarrow e_{01,2}^{A},
\]

因此 X11/X12 是固定深度、固定参数量的因果对照。

数据必须包含 matched interventions：固定

\[
(x_0,x_1,O_0,O_1)
\]

以及 edge ((0,1)) 可见的全部输入，只独立改变 ((x_2,O_2))。同一个 base pair 的所有 intervention 必须整体进入同一 split，不能跨 train/test 泄漏。
除预测 Jacobian 是否非零外，还要比较预测与真值的 cross-Jacobian（或相同方向扰动下的有限差分）误差与相关性；否则“非零”本身不足以证明学对了三体响应。

## 5. 统一训练与报告协议

由外部 runner 实现，但必须固定以下口径：

- X03–X12 使用相同的 split 生成规则；成对对照必须使用逐样本相同的 split；
- X03/X04、X05/X06、X09/X10、X11/X12 必须从完全相同的初始 state 开始，并使用相同步数、batch 顺序和模型选择规则；报告 paired delta、置信区间和效应量；
- 至少 5 个 model seed，分别报告 median 与离散度，不能只报最佳 seed；
- model seed、split seed、shuffle seed 分开记录；
- early stopping、训练预算和模型选择指标在比较前固定；
- 报 node component MAE/RMSE、vector MAE、NMSE；有 pair 标签时同时报 pair 指标；
- 报参数量、wall time、peak memory、每层 (B) norm 和 gradient norm；
- X02 先验证 OPLS pair labels 经 signed scatter 后与 node labels 数值闭合，再报 wrapper 与显式 pair loop 的最大绝对/相对误差；整个实验固定 checkpoint、使用 eval mode 并冻结 RMS；
- X09/X10 固定每个 (N) 的几何密度/采样规则和训练样本权重，按节点数分别报告 per-node 指标，不把所有 (N) 混成一个平均数；
- X11/X12 除 test error 外，还报
  \(\left\|\partial e_{01,L}^{A}/\partial x_2\right\|\)。

RunningRMS 规则：训练期更新，validation/test 冻结；独立层各自保存 RMS。
`shared_all` 会共享 Message Block、外部 node-update (P) 和 RunningRMS，并在一个 forward 中重复使用；它只能解释为 recurrent 参数效率/工程对照，不能称作隔离了纯 weight sharing 的因果实验。

当前 core 支持动态 (N,E)，但同一 batch 内必须共享节点数和 `pair_index`。
X09/X10 应按节点数/topology 分 bucket；不要伪装成 ragged graph 支持。

## 6. 每个模型都必须通过的数学审计

以下不是额外调参实验，而是所有 12 项的验收门：

1. 全局平移不变与世界坐标旋转协变；
2. receiver / sender 独立 (D_6) gauge；
3. 节点重标记与 edge-list 顺序等变；
4. 对每条无向边交换 `pair_index` 两端后，force 与 edge (A) 按规定的方向/符号对应；该审计必须覆盖多层传播；
5. 每条无向边只计算一次；
6. \(\left\|\sum_iF_i\right\|\) 达到数值精度；
7. `none`、`sum`、`mean` 在单层最终 force 上应完全相同，因为聚合只在层间生效；
8. 多层中所有层使用相同 `pair_index` 顺序，保证 edge (A) state 对齐；
9. eval 阶段 RunningRMS 不再变化；
10. 每次运行记录实际编译的 `path_manifest` SHA-256，并与 X01 reference fingerprint 相同；只比较手写 config 不足以证明 path bank 未改变。

## 7. 合理性审核结论

该设计是合理的，原因如下：

- **变量可归因。** X05/X06 只改变 (B) 聚合，深度、参数量、edge-(A) 传播完全相同；因此能够单独归因跨边通信。
- **负对照与正对照齐全。** OPLS 是 pair-additive 负对照；三体链是聚合必需的正对照。仅在 OPLS 上提升不能被误写成“学到了多体力”。
- **表达能力与训练困难分开。** X03/X04 区分 node-force credit assignment 和 edge 表达上限。
- **聚合律比较可辨识。** X09/X10 使用 degree shift；固定五节点完全图上的 sum/mean 对照基本只是可吸收的常数缩放，不足以形成结论。
- **没有偷偷扩内核。** learnable node update 只在相同 TypeKey 内混 channel；path、Gate 和 covariant basis 均未改变。
- **多体证据不只靠 loss。** X11/X12 同时检查性能和结构 Jacobian，避免数据泄漏或偶然相关造成假阳性。

当前刻意不研究 invariant energy readout、严格总转矩、混合分子类型和真实 ab-initio 多体数据。
这些会改变任务定义，适合作为本轮图状态流通过之后的下一阶段，而不应混入本轮 Message Block 验收。

## 8. 核心代码边界

配套文件 `e311_gnn_12_experiment_core_v1.py` 只实现：

- 12 项实验的不可变 registry；
- 动态 complete-graph `pair_index` helper；
- receiver-local `none/sum/mean` 聚合；
- 无 bias 的 same-TypeKey node update；
- 1–4 层、independent / `shared_all` Message Block stack；
- 最后一层 force readout、状态输出与 architecture ledger。

ledger 同时记录 dtype、force scale、STF ranks、TypeKeys、实际 path 数量与 `path_manifest` SHA-256，防止依赖源码变化后仍误称为同一冻结内核。

数据接口、loss、optimizer、scheduler、checkpoint loading、CLI、logging 和并行化全部留给后续 runner，不在本文件中预设。
