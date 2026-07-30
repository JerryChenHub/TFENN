# TFENN

TFENN 使用预先确定的 Tensor Basis 和有限群平均构造神经网络，用于预测双苯体系中的合力与合力矩。

## 对称约定

群包含十二个三维 proper rotations：

```text
G = {Rz(kπ/3), Rx(π)Rz(kπ/3)},  k = 0,...,5
```

代码存储行向量。对于任意 `g1, g2 ∈ G`，输入与输出满足：

```text
x_new = x @ g1
R_new = g1.T @ R @ g2
model(x_new, R_new) = model(x, R) @ g1
```

因此输出对左侧苯分子的作用协变，对右侧苯分子的作用不变。输出形状为 `(batch, 2, 3)`，通道 `0` 是力，通道 `1` 是力矩。当前研究不编码两个苯分子互换的对称性。

## 模型

`D6TensorBasisNetV1` 使用 Tensor Basis 分支，以及轻量的 Tensor Basis 到向量映射和右群平均。

`D6TensorBasisNetV2` 保留 Tensor Basis 分支，但在 Tensor 到向量阶段使用完整双群平均。

`D6GroupAverageNetV1` 的主要映射均由显式群平均构造，用于方法对比。

`D6SymmetrizedMLPBaselineV1` 对普通 MLP 做完整输入输出群投影。

`MLPBaselineV1` 是不带结构对称约束的普通基线。

## 目录

```text
src/TFENN/nn                 Layer 与 Gate
src/TFENN/symmetry           D6 群和表示作用
src/TFENN/models.py          Option1 模型
src/TFENN/data               数据读取与生成
data/benzene_pair            数据及同名参数 JSON
experiments/benzene_pair     单一训练入口和实验日志
tests                        数学性质与代码测试
```

## Windows 环境

项目使用已有的 `ml_torch` Conda 环境：

```powershell
D:\miniconda3\Scripts\conda.exe run -n ml_torch python -m pip install --no-deps --editable .
D:\miniconda3\Scripts\conda.exe run -n ml_torch python -m pytest
```

实际环境版本记录在 `windows_versions.json`。

## 训练

所有网络共用一个训练入口，并通过 `model_name` 选择模型：

```powershell
D:\miniconda3\Scripts\conda.exe run -n ml_torch python experiments\benzene_pair\train.py --model_name D6TensorBasisNetV1 --experiment_name first_test
```

每次实验在 `experiments/benzene_pair/logs` 中写入一个 JSON。日志包含模型结构、数据生成参数、优化器、损失、训练参数、loss 历史、环境版本和完整 144 组 D6 作用的误差。已有同名日志不会被覆盖。

## 数据生成

```python
from TFENN.data import (
    BenzenePairGenerationConfig,
    generate_benzene_pair_dataset,
)

config = BenzenePairGenerationConfig(
    sample_count=10_000,
    seed=7,
    distance_range=(6.0, 10.0),
    min_separation=4.0,
)
generate_benzene_pair_dataset("data/benzene_pair/example.csv", config)
```

生成器同时写入 `example.csv` 和 `example.json`。JSON 保存精简的生成参数，CSV 只保存训练所需数值。
