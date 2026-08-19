# Notation for Invariant-Gated Covariant Activation

## 1. Scope

This document fixes the notation used by the generic activation layer. It separates four distinct objects:

1. representation types and their live feature vectors;
2. polynomial or tensor-product source lifts;
3. frozen equivariant maps compiled for a registered path;
4. trainable invariant gates and channel mixing.

The canonical channel-free activation is

\[
\boxed{
\operatorname{Act}_{\tau\leftarrow\Sigma}
(z_1,\ldots,z_k;\mathbf{s})
=
\sum_{\alpha=1}^{n_{\tau\leftarrow\Sigma}}
\gamma^{\tau}_{\Sigma,\alpha}(\mathbf{s})
C_{\tau\leftarrow\Sigma,\alpha}
\Lambda_\Sigma(z_1,\ldots,z_k).
}
\]

The label \(\tau\leftarrow\Sigma\) is read as “from source signature \(\Sigma\) to target type \(\tau\).”

## 2. Representation types

Let \(G\) be the symmetry group and let \(\mathscr T\) be the registered set of representation types.

For each \(\sigma\in\mathscr T\):

| Symbol | Meaning |
|---|---|
| \(V_\sigma\) | Real representation space of type \(\sigma\) |
| \(d_\sigma=\dim V_\sigma\) | Representation dimension |
| \(\rho_\sigma(g)\) | Matrix representing \(g\in G\) on \(V_\sigma\) |
| \(z\in V_\sigma\) | One live feature vector of type \(\sigma\) |

The group action is

\[
z\longmapsto \rho_\sigma(g)z.
\]

The symbol \(A\) denotes the registered A-type representation. The symbol \(B^{(r)}\) denotes a registered B-type block with block label \(r\). The label \((r)\) is not a tensor power.

## 3. Source signature

A source signature is

\[
\boxed{
\Sigma=((\sigma_1,p_1),\ldots,(\sigma_k,p_k)).
}
\]

| Symbol | Meaning |
|---|---|
| \(\Sigma\) | Complete ordered source signature |
| \(k\) | Number of independent live input slots |
| \(t\in\{1,\ldots,k\}\) | Slot index |
| \(\sigma_t\) | Representation type of slot \(t\) |
| \(p_t\in\mathbb N_{\ge 1}\) | Number of times the same live vector \(z_t\) is used in that slot |
| \(z_t\in V_{\sigma_t}\) | Runtime value supplied to slot \(t\) |

The order and names of independent slots are part of the registered signature.

Two cases must not be confused:

\[
\Sigma=((A,2))
\]

contains one live vector \(a\), repeated twice as \(a^{\otimes 2}\). In contrast,

\[
\Sigma=((B^{(i)},1),(B^{(j)},1))
\]

contains two independent live vectors, even when \(i=j\).

## 4. Single-slot polynomial lift

For \(z\in V_\sigma\) and power \(p\), define

\[
D_{\sigma,p}
=
\dim\operatorname{Sym}^p(V_\sigma)
=
\binom{d_\sigma+p-1}{p}.
\]

Let

\[
U_{d_\sigma,p}
\in
\mathbb R^{d_\sigma^p\times D_{\sigma,p}}
\]

contain an orthonormal basis of the symmetric subspace. Its columns satisfy

\[
U_{d_\sigma,p}^{\top}U_{d_\sigma,p}=I_{D_{\sigma,p}}.
\]

The normalized polynomial lift of one slot is

\[
\boxed{
\lambda_{\sigma,p}(z)
=
U_{d_\sigma,p}^{\top}z^{\otimes p}
\in\mathbb R^{D_{\sigma,p}}.
}
\]

For \(p=1\), use \(U_{d_\sigma,1}=I_{d_\sigma}\), so

\[
\lambda_{\sigma,1}(z)=z.
\]

For \(a=(x,y,z)^{\top}\in\mathbb R^3\), one normalized \(p=2\) coordinate convention is

\[
\lambda_{A,2}(a)
=
\begin{pmatrix}
x^2 & \sqrt2xy & \sqrt2xz & y^2 & \sqrt2yz & z^2
\end{pmatrix}^{\top}
\in\mathbb R^6.
\]

The exact coordinate order is fixed by the registered \(U\) basis and must match the offline compiler.

The induced symmetric-power representation is

\[
S_{\sigma,p}(g)
=
U_{d_\sigma,p}^{\top}
\rho_\sigma(g)^{\otimes p}
U_{d_\sigma,p}
\in\mathbb R^{D_{\sigma,p}\times D_{\sigma,p}},
\]

and the lift transforms as

\[
\lambda_{\sigma,p}(\rho_\sigma(g)z)
=
S_{\sigma,p}(g)\lambda_{\sigma,p}(z).
\]

Thus, a polynomial lift is covariant, not invariant.

## 5. Complete source lift

The lifted source space for \(\Sigma\) is

\[
W_\Sigma
=
\bigotimes_{t=1}^{k}
\operatorname{Sym}^{p_t}(V_{\sigma_t}).
\]

Its dimension is

\[
\boxed{
D_\Sigma
=
\dim W_\Sigma
=
\prod_{t=1}^{k}D_{\sigma_t,p_t}.
}
\]

The complete source lift is

\[
\boxed{
\Lambda_\Sigma(z_1,\ldots,z_k)
=
\bigotimes_{t=1}^{k}
\lambda_{\sigma_t,p_t}(z_t)
\in\mathbb R^{D_\Sigma}.
}
\]

Lowercase \(\lambda\) acts on one slot. Uppercase \(\Lambda\) combines all slots in the signature.

The induced representation on \(W_\Sigma\) is

\[
\rho_\Sigma(g)
=
\bigotimes_{t=1}^{k}S_{\sigma_t,p_t}(g).
\]

## 6. Target type and frozen covariant basis

The target type is \(\tau\in\mathscr T\), with output space \(V_\tau\), dimension \(d_\tau\), and action \(\rho_\tau(g)\).

The legal linear maps from the lifted source to the target form the intertwiner space

\[
\operatorname{Hom}_G(W_\Sigma,V_\tau).
\]

Its dimension is

\[
\boxed{
n_{\tau\leftarrow\Sigma}
=
\dim\operatorname{Hom}_G(W_\Sigma,V_\tau).
}
\]

The index

\[
\alpha=1,\ldots,n_{\tau\leftarrow\Sigma}
\]

labels a frozen basis of this space:

\[
C_{\tau\leftarrow\Sigma,\alpha}
\in
\mathbb R^{d_\tau\times D_\Sigma}.
\]

Every basis matrix satisfies

\[
\boxed{
C_{\tau\leftarrow\Sigma,\alpha}\rho_\Sigma(g)
=
\rho_\tau(g)C_{\tau\leftarrow\Sigma,\alpha}.
}
\]

The symbol \(\alpha\) is an intertwiner-basis index. It is not a channel, layer, molecule, or B-block index.

When \(n_{\tau\leftarrow\Sigma}=0\), the path does not exist and no activation module should be constructed for it.

## 7. Invariants and gates

Let

\[
\mathbf{s}\in\mathbb R^{d_s}
\]

be the invariant descriptor used by the gate. Bold \(\mathbf{s}\) avoids confusion with B-block labels. It satisfies

\[
\mathbf{s}\longmapsto\mathbf{s}
\]

under the common group action.

For a channel-free path,

\[
\gamma^\tau_\Sigma(\mathbf{s})
\in
\mathbb R^{n_{\tau\leftarrow\Sigma}},
\]

and

\[
\gamma^{\tau}_{\Sigma,\alpha}(\mathbf{s})\in\mathbb R
\]

is the runtime coefficient of basis element \(\alpha\). The value \(\gamma\) is not itself a stored parameter; it is produced by a trainable gate network.

Separate the scalar Gate trunk from its coefficient head. Write \(v\) for a
DAG stage and \(\ell(v)\) for its execution level. Parallel stages at the same
level may have different Gate widths. At stage \(v\),
write the model-declared scalar network as

\[
h_v^{\mathrm{gate}}(\mathbf s)
=
\operatorname{GateTrunk}_{v,\theta_v}(\mathbf s)
\in\mathbb R^{w_v^{\mathrm{gate}}},
\]

where its depth and hidden activation must be stated by the model. The current
one-hidden-layer reference specializes this to

\[
h_v^{\mathrm{gate}}(\mathbf s)
=\phi_v(W_{s,v}\mathbf s+b_{s,v}).
\]

It is followed, for each registered path
\(\kappa=(\tau\leftarrow\Sigma,\text{source roles})\), by

\[
u_{v,\kappa}(\mathbf s)
=W_{\gamma,v,\kappa}h_v^{\mathrm{gate}}(\mathbf s)
+b_{\gamma,v,\kappa},
\qquad
\gamma_{v,\kappa}(\mathbf s)
=\psi_{v,\kappa}(u_{v,\kappa}(\mathbf s)).
\]

Here \(w_v^{\mathrm{gate}}\) is the scalar Gate hidden width. The notation
\(w_\ell^{\mathrm{gate}}\) is reserved for a width tied across all stages at
level \(\ell\). The current
signed coefficient convention is

\[
\psi_{v,\kappa}=\operatorname{id},
\qquad
\gamma_{v,\kappa}=u_{v,\kappa}.
\]

Identity means that the final coefficient is not passed through sigmoid,
tanh, softplus, or another range restriction. It does not mean
\(\gamma=1\). A linear coefficient readout does not make the complete
activation linear: the descriptor encoder, polynomial lift, multiplicative
gate, and composition of stages remain separate sources of nonlinearity.

## 8. Channel notation

Representation coordinates and feature channels are different axes.

For slot \(t\), the runtime tensor has shape

\[
Z_t\in\mathbb R^{L_1\times\cdots\times L_m\times c_t\times d_{\sigma_t}},
\]

abbreviated as

\[
Z_t:\quad(*\mathcal L,c_t,d_{\sigma_t}).
\]

| Symbol | Meaning |
|---|---|
| \(*\mathcal L\) | Shared leading axes, such as batch, node, or edge axes |
| \(c_t\) | Number of feature channels for input slot \(t\) |
| \(c_o\) | Number of output feature channels |
| \(r_{\mathrm{mix}}\) | Latent channel-mixing rank in an optional factorized path |
| \(q\in\{1,\ldots,r_{\mathrm{mix}}\}\) | Latent channel index |
| \(P^{(t)}\in\mathbb R^{r_{\mathrm{mix}}\times c_t}\) | Trainable input-channel projection for slot \(t\) |
| \(P_{\mathrm{out}}\in\mathbb R^{c_o\times r_{\mathrm{mix}}}\) | Trainable output-channel projection |

The latent mixing rank \(r_{\mathrm{mix}}\) is the number of learned latent
channel combinations evaluated for one factorized covariant path. It controls
the capacity and cost of the factorized channel parameterization before the
fixed representation-space map is applied. It is not a runtime-estimated
algebraic rank, group rank, STF rank, representation dimension,
polynomial degree, typed channel count, or Gate hidden width.

Channel mixing must be named explicitly:

| Mode | Coefficient/channel structure |
|---|---|
| `dense` | A coefficient is produced for every output channel, source-channel tuple, and intertwiner basis element. No latent mixing rank is used. |
| `factorized(r_mix)` | Input channels are projected to \(r_{\mathrm{mix}}\) latent combinations, the covariant path is evaluated in that latent index, and \(P_{\mathrm{out}}\) maps to output channels. |
| `diagonal` | Only explicitly aligned source-channel combinations are used. It is a restricted model and must not be described as full channel mixing. |

In dense mode, the path coefficient tensor has shape

\[
\gamma_{v,\kappa}(\mathbf s)
\in
\mathbb R^{c_o\times c_1\times\cdots\times c_k
\times n_{\tau\leftarrow\Sigma}}.
\]

Thus \(\gamma_{v,\kappa}\) denotes the whole coefficient tensor for one
stage/path. In dense mode its components are
\(\gamma_{v,\kappa,c,c_1,\ldots,c_k,\alpha}\); the shorter
\(\gamma^\tau_{\Sigma,\alpha}\) is a stage/channel-suppressed component
notation. In factorized mode, channel tuples are replaced by latent index
\(q\), giving \(\gamma_{v,\kappa,q,\alpha}\).

The current E311/F reference uses dense gamma heads and a dense
representation-preserving output-channel projection. The following equations
define the optional factorized mode.

The projected live vector is

\[
\widetilde z_{t,q}
=
\sum_{c=1}^{c_t}P^{(t)}_{q,c}Z_{t,c}
\in V_{\sigma_t}.
\]

For each latent channel and covariant basis element,

\[
v_{q,\alpha}
=
C_{\tau\leftarrow\Sigma,\alpha}
\Lambda_\Sigma(
\widetilde z_{1,q},\ldots,\widetilde z_{k,q})
\in V_\tau.
\]

The factorized activation is

\[
\boxed{
y_c
=
\sum_{q=1}^{r_{\mathrm{mix}}}(P_{\mathrm{out}})_{c,q}
\sum_{\alpha=1}^{n_{\tau\leftarrow\Sigma}}
\gamma_{v,\kappa,q,\alpha}(\mathbf{s})
v_{q,\alpha},
\qquad c=1,\ldots,c_o.
}
\]

Its output shape is

\[
Y:\quad(*\mathcal L,c_o,d_\tau).
\]

The extra index \(q\) extends the channel-free gate coefficient to the factorized channel model.

## 9. Representative signatures

### 9.1 \(A\to A\)

\[
\tau=A,
\qquad
\Sigma=((A,1)),
\qquad
\Lambda_\Sigma(a)=a.
\]

Therefore

\[
D_\Sigma=d_A,
\qquad
C_{A\leftarrow((A,1)),\alpha}
\in\mathbb R^{d_A\times d_A}.
\]

### 9.2 Quadratic \(A\to B^{(r)}\)

\[
\tau=B^{(r)},
\qquad
\Sigma=((A,2)),
\]

\[
\Lambda_\Sigma(a)
=
\lambda_{A,2}(a)
=
U_{d_A,2}^{\top}a^{\otimes2}.
\]

Therefore

\[
D_\Sigma=\binom{d_A+1}{2},
\qquad
C_{B^{(r)}\leftarrow((A,2)),\alpha}
\in
\mathbb R^{d_{B^{(r)}}\times D_\Sigma}.
\]

For \(d_A=3\), \(D_\Sigma=6\).

### 9.3 \(B^{(i)}\times B^{(j)}\to B^{(r)}\)

\[
\tau=B^{(r)},
\qquad
\Sigma=((B^{(i)},1),(B^{(j)},1)),
\]

\[
\Lambda_\Sigma(b^{(i)},b^{(j)})
=
b^{(i)}\otimes b^{(j)}.
\]

Therefore

\[
D_\Sigma=d_{B^{(i)}}d_{B^{(j)}},
\qquad
C_{B^{(r)}\leftarrow\Sigma,\alpha}
\in
\mathbb R^{d_{B^{(r)}}\times D_\Sigma}.
\]

## 10. Lifecycle classification

| Object | Status | Role |
|---|---|---|
| \(U_{d_\sigma,p}\) | Frozen buffer | Defines the normalized symmetric basis |
| \(C_{\tau\leftarrow\Sigma,\alpha}\) | Frozen buffer | Defines a legal covariant direction |
| \(P^{(t)}\), \(P_{\mathrm{out}}\) | Trainable | Mix input and output channels in the optional factorized mode |
| Gate encoder/head parameters | Trainable | Produce invariant coefficients |
| Descriptor RMS statistics | Training-calibrated frozen buffer | Normalize a fixed invariant schema without becoming a learned feature |
| \(Z_t\), \(\mathbf{s}\) | Runtime data | Supply live covariants and invariants |
| \(\gamma(\mathbf{s})\) | Runtime value | Weights frozen covariant basis outputs |

## 11. Meaning of “polynomial”

For fixed \(C_{\tau\leftarrow\Sigma,\alpha}\),

\[
C_{\tau\leftarrow\Sigma,\alpha}
\Lambda_\Sigma(z_1,\ldots,z_k)
\]

is polynomial in the live covariant inputs, with total degree

\[
\sum_{t=1}^{k}p_t.
\]

If \(\gamma(\mathbf{s})\) is produced by a nonlinear MLP, the complete gated activation is generally not a polynomial in all underlying model features. “Polynomial path” refers specifically to the frozen \(C\Lambda\) part.

## 12. Covariant flow versus invariant reuse

The activation formula does not require one universal network graph. Every
model must separately declare:

1. a covariant-flow policy:
   `STRICT_DECLARED_FLOW`, `EXPLICIT_REUSE_FLOW`, or `DENSE_HISTORY_FLOW`;
2. a raw-covariant-access policy:
   `STEM_ONLY_RAW`, `EXPLICIT_RAW_REREAD`, or `EVERY_STAGE_RAW`;
3. an invariant-context policy:
   `RAW_ONLY_INVARIANTS`, `RAW_PLUS_PARENTS_INVARIANTS`,
   `DENSE_HISTORY_INVARIANTS`, or `EXPLICIT_INVARIANTS` with an explicit
   invariant source/signature list.

The invariant-context declaration has two parts: source scope and enabled
scalar families. List radial/raw, unary, symmetric-power, raw-hidden mixed,
hidden-hidden mixed, and STF shortcuts as applicable; a scope label such as
`FULL` is not sufficient.

Strict flow means that live covariant tensors follow only declared graph
edges. It is not mandatory for every TFENN model. Reusing scalar invariants at
several Gates does not create a covariant edge. Likewise, reading an earlier
typed stage again under an explicit reuse policy is legal, but it must be
listed rather than hidden in an implementation default.

The policy name `NO_RAW_MIXED` must be qualified by bank. For example,
`NO_RAW_MIXED_COVARIANTS` may disable raw-mixed covariant output branches
while raw-hidden invariant contractions remain enabled.

## 13. Required path and Gate manifest

A model description must state the actual mixture of path families rather
than only say “A/B mixing.” At minimum, record:

| Part | Required declaration |
|---|---|
| \(A\to A\) | enabled unary, self-symmetric, independent-slot, or mixed signatures |
| \(A\to B^{(r)}\) | linear unary \((A,1)\), pure-A \((A,p)\), and any mixed A/B signatures |
| \(B^{(s)}\to B^{(r)}\) | linear cross-block unary paths and any self-symmetric paths |
| \(B\times B\to B\) | the independent B slots and target blocks; do not substitute a symmetric self-power |
| \(B\to A\) / direct A head | enabled unary, symmetric, independent-slot, and mixed signatures |
| channel mixing | `dense`, `factorized(r_mix)`, or `diagonal` |
| same-type flow | bypass sources and output channel projection |
| Gate | invariant-context policy, trunk depth/width/activation, head parameterization, and coefficient-output activation |

All path existence and basis multiplicities come from the compiled registered
Hom spaces for the supplied group. A model configuration selects from those
legal paths; it must not infer or hard-code them from the benzene example.
