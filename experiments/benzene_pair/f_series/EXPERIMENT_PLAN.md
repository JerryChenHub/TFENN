# F Series: strict typed-flow channel study

## Notation and fixed mathematics

- `a2:A[c]` means node `a2` contains `c` copies of the A representation,
  with runtime shape `[..., c, 3]`. The digit in `a2` is a node index, not a
  channel count.
- `b1:B[c]` means `c` channels **for each registered B TypeKey**. For benzene
  proper-D6 this gives separate rank-2 and rank-6 blocks with shapes
  `[..., c, 5]` and `[..., c, 13]`.
- `W8` is the hidden width of the scalar Invariant-Gate trunk. It is not an
  A/B channel count.
- Raw covariant ingress is fixed: `x -> a1:A` and
  `R -> PoseEncoder -> r -> b1:B`. Raw pose never enters through A.
- Covariant tensors can only follow the declared graph edges. A same-TypeKey
  edge always retains the legacy bypass and is jointly channel-projected with
  its gated branches. A cross-TypeKey edge is implemented only by compiled
  unary and `Sym^2` intertwiners.
- There are no undeclared raw/history covariant rereads, no hidden-hidden
  tensor-product paths, no degree-3 paths, and no group convolution.
- The E311 Gate mechanics are fixed: one-layer SiLU trunk, `W=8`, dense gamma
  heads, signed linear coefficient output, unary plus `Sym^2` covariants,
  NO_RAW_MIXED covariant pair bank, legacy same-type bypass, and dense channel
  projection.

## Strict topologies

| Topology | Declared covariant edges |
|---|---|
| T1 | `x->a1`, `r->b1`, `a1,b1->a2`, `b1,a1->b2`, `a2,b2->a3`, `a3->out` |
| T2 | `x->a1`, `r->b1`, `a1,b1->a2`, `a2->a3`, `a3->out` |
| T3 | `x->a1`, `r->b1`, `a1->a2`, `b1->b2`, `a2,b2->a3`, `a3->out` |

Stages at the same execution level are synchronous and read the frozen state
from the previous level.

## Scientific experiments

- `F100`: exact historical E311 control; it is not a strict-flow model.
- `F1 = F101..F150` (`RAW_LOCAL_MIX`): at every Gate,
  `invariant_source_names = {x, r} union declared hidden parents`. The full
  local descriptor contains the raw core, parent unary invariants, parent
  `Sym^2` invariants, and raw-parent scalar contractions. It contains no
  undeclared hidden history and no hidden-hidden pair invariant.
- `F2 = F201..F250` (`RAW_ONLY_MASK`): exact paired copy of F1 with the same
  compiled descriptor schema, trainable shapes, initialization, covariant
  paths, and parameter count. A fixed mask retains only radial/raw descriptors
  and zeroes every hidden-derived descriptor column.

Thus F1 versus F2 tests the value of **all local hidden-derived invariant
context**, not only the raw-hidden pair subset.

The 50 strict structures are one-node-at-a-time channel sweeps:

- T1 and T3: baseline `A=(1,1,1)`, `B=(2,1)`; vary A1 over `2,4,8`, A2 over
  `2,4,8,16`, A3 over `2,4,8`, B1 over `1,4,8`, and B2 over `2,4,8`.
- T2: baseline `A=(1,1,1)`, `B1=2`; vary each A node over `2,4,8,16` and B1
  over `1,4,8`.

## Execution shards (organization only)

The five tmux sessions do not define five scientific experiments:

| tmux session | Models |
|---|---|
| `tfenn_f_control` | F100 |
| `tfenn_f1_a` | F101-F125 |
| `tfenn_f1_b` | F126-F150 |
| `tfenn_f2_a` | F201-F225 |
| `tfenn_f2_b` | F226-F250 |

Prepare the exact E311 split and compile all 101 models once, then launch:
`COMET_API_KEY` must be exported in the environment used by the tmux server.

```bash
python -m experiments.benzene_pair.f_series.runner prepare \
  --reference_split_directory <E311_SHARED_SPLIT>
python -m experiments.benzene_pair.f_series.runner launch-tmux \
  --device cuda:0 --device cuda:1 --device cuda:2 \
  --device cuda:3 --device cuda:4
```

With no `--device`, the launcher automatically maps the first five visible
CUDA devices and fails closed if fewer than five are visible. Pass one
`--device` only to share it intentionally, or repeat `--device` exactly five
times. After completion:

```bash
python -m experiments.benzene_pair.f_series.runner aggregate
```

Every selected checkpoint records the complete model checkpoint, a dedicated
role-labelled Invariant-Gate parameter artifact, descriptor/head magnitude
summaries, validation gamma statistics, and pre-projection branch RMS. The
primary comparison metric is Final Test MAE; D6 covariance must pass.
