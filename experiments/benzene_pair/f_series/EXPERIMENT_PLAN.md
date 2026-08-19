# F Series: strict typed-flow channel study

Strict flow is fixed for F101--F250 only; it is not a project-wide topology
requirement.

## Notation and fixed mathematics

- `a2:A[c]` means node `a2` contains `c` copies of the A representation,
  with runtime shape `[..., c, 3]`. The digit in `a2` is a node index, not a
  channel count.
- `b1:B[c]` means `c` channels **for each registered B TypeKey**. For the
  current benzene `D6~` compiler output this gives separate rank-2 and rank-6 blocks with shapes
  `[..., c, 5]` and `[..., c, 13]`.
- `W8` means \(w_v^{\mathrm{gate}}=8\) at every stage. It is not an A/B
  channel count or latent mixing rank.
- Raw covariant ingress is fixed: `x -> a1:A` and
  `R -> PoseEncoder -> r -> b1:B`. Raw pose never enters through A.
- The study uses `STRICT_DECLARED_FLOW` with `STEM_ONLY_RAW`: covariant tensors
  can only follow the declared graph edges. A same-TypeKey
  edge retains the declared concat-and-project bypass and is jointly channel-projected with
  its gated branches. A cross-TypeKey edge is implemented only by compiled
  unary and `Sym^2` intertwiners.
- There are no undeclared raw/history covariant rereads, no hidden-hidden
  tensor-product paths, no degree-3 paths, and no group convolution.
- The E311 Gate mechanics are fixed: one-layer SiLU trunk, `W=8`, dense gamma
  heads, signed unbounded identity coefficient output, unary plus `Sym^2` covariants,
  `NO_RAW_MIXED_COVARIANTS`, declared same-TypeKey bypass, and dense channel
  mixing/projection.

## Strict topologies

| Topology | Declared covariant edges |
|---|---|
| T1 | `x->a1`, `r->b1`, `a1,b1->a2`, `b1,a1->b2`, `a2,b2->a3`, `a3->out` |
| T2 | `x->a1`, `r->b1`, `a1,b1->a2`, `a2->a3`, `a3->out` |
| T3 | `x->a1`, `r->b1`, `a1->a2`, `b1->b2`, `a2,b2->a3`, `a3->out` |

Stages at the same execution level are synchronous and read the same
pre-update state snapshot.

## Scientific experiments

- `F100`: exact historical E311 control using `DENSE_HISTORY_FLOW`,
  `EVERY_STAGE_RAW`, and `DENSE_HISTORY_INVARIANTS`; it is not a strict-flow
  model.
- `F1 = F101..F150` (`RAW_LOCAL_MIX`, implementing
  `RAW_PLUS_PARENTS_INVARIANTS`): at every Gate,
  `invariant_source_names = {x, r} union declared hidden parents`. The full
  local descriptor contains the raw core, parent unary invariants, parent
  `Sym^2` invariants, and raw-parent scalar contractions. It contains no
  undeclared hidden history and no hidden-hidden pair invariant.
- `F2 = F201..F250` (`RAW_ONLY_MASK`, implementing the active
  `RAW_ONLY_INVARIANTS` context): exact paired copy of F1 with the same
  compiled descriptor schema, trainable shapes, initialization, covariant
  paths, and parameter count. A fixed mask retains only radial/raw descriptors
  and zeroes every hidden-derived descriptor column.

Thus F1 versus F2 tests the value of **all local hidden-derived invariant
context**, not only the raw-hidden pair subset.

The 50 strict structures are one-node-at-a-time channel sweeps:

- T1 and T3 baseline: `a1:A[1]`, `a2:A[1]`, `a3:A[1]`, `b1:B[2]`,
  `b2:B[1]`. Vary `a1:A[c]` over `c=2,4,8`, `a2:A[c]` over
  `c=2,4,8,16`, `a3:A[c]` over `c=2,4,8`, `b1:B[c]` over `c=1,4,8`, and
  `b2:B[c]` over `c=2,4,8`.
- T2 baseline: `a1:A[1]`, `a2:A[1]`, `a3:A[1]`, `b1:B[2]`. Vary each
  `a*:A[c]` over `c=2,4,8,16` and `b1:B[c]` over `c=1,4,8`.

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
primary comparison metric is Final Test MAE; \(D_6^{\mathrm{rot}}\) (`D6~`)
covariance must pass.
