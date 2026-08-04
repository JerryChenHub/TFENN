## Mathematical Notation and Naming Convention

Use the notation in this section consistently throughout mathematical derivations, documentation, code comments, and architecture descriptions. Do not introduce alternative symbols for the same objects.

This convention is molecule-independent. Never specialize the notation to a particular molecular species.

### 1. Global restrictions

- Use exactly one root molecular body frame for the entire molecular cloud.
- Use molecule `0` as the root unless another root is explicitly specified.
- Never introduce node-local or edge-local coordinate frames.
- Never use group averaging.
- Never use `Q` or `T` to denote components of the B representation.
- Denote B components by `B^{(r)}`, with the component label enclosed in parentheses.
- A superscript such as `B^{(r)}` is a type label, not a tensor power.
- Always write tensor powers explicitly as `v^{\otimes p}`.
- Put node indices and layer indices in subscripts.
- Reserve uppercase `F_i` for world-frame forces and lowercase `f_i` for root-frame forces.

### 2. World-frame and root-frame geometry

For molecule \(i\), use

\[
x_i
\]

for its world-frame center and

\[
O_i
\]

for its world-frame orientation.

Use \(O_i\), not \(R_i\), for a world-frame orientation.

Define the root-frame position and relative orientation by

\[
X_i = O_0^\top(x_i-x_0),
\]

\[
R_i = O_0^\top O_i.
\]

Thus:

- \(x_i\): world-frame position;
- \(O_i\): world-frame orientation;
- \(X_i\): position in the single root body frame;
- \(R_i\): orientation relative to the root body frame.

For an ordered molecular pair \(j\to i\), define

\[
d_{ij}=X_j-X_i.
\]

Never use \(d_{ij}\) to denote a distance scalar. Squared distance must be written as

\[
d_{ij}^\top d_{ij}.
\]

### 3. A/B representation types

Let the set of feature types be

\[
\mathscr T
=
\left\{
A,
B^{(r)}:r\in\mathcal R_B
\right\},
\]

where \(\mathcal R_B\) indexes the B components required by the molecular representation.

When there are two B components, write

\[
\mathscr T
=
\left\{
A,B^{(1)},B^{(2)}
\right\}.
\]

Use

\[
\rho_\tau(g)
\]

for the group action on type \(\tau\in\mathscr T\).

An object \(z^\tau\) has type \(\tau\) if

\[
z^\tau\mapsto\rho_\tau(g)z^\tau.
\]

In particular,

\[
a\mapsto\rho_A(g)a
\]

for an A-type object, and

\[
b^{(r)}
\mapsto
\rho_{B^{(r)}}(g)b^{(r)}
\]

for a \(B^{(r)}\)-type object.

All node features share the same root action \(g_0\). No node-dependent representation action may be introduced after root-frame preprocessing.

### 4. B encoding

Use

\[
\overline B^{(r)}
\]

for a fixed body-frame template of B component \(r\). The overline always means that the object is fixed in the canonical molecular body frame.

Use

\[
B_i^{(r)}
\]

for the corresponding B descriptor of molecule \(i\).

When the descriptor is generated from a relative orientation, write

\[
B_i^{(r)}
=
\beta^{(r)}(R_i).
\]

It must satisfy

\[
\beta^{(r)}(g_0R_i g_i)
=
\rho_{B^{(r)}}(g_0)\beta^{(r)}(R_i).
\]

Consequently,

\[
\beta^{(r)}(R_i g_i)
=
\beta^{(r)}(R_i).
\]

The sender action \(g_i\) is therefore absent from every downstream computation.

When a fixed tensor template is used, the encoding may be written as

\[
B_i^{(r)}
=
\rho_{B^{(r)}}(R_i)\overline B^{(r)}.
\]

Do not use `Q_i`, `T_i`, or species-specific names as replacements for \(B_i^{(r)}\).

### 5. Hidden features

Use

\[
h_{i,\ell}^{\tau}
\]

for the type-\(\tau\) hidden feature of molecule \(i\) at network layer \(\ell\).

The indices have fixed meanings:

- \(i\): molecular node;
- \(\ell\): network layer;
- \(\tau\): representation type.

Examples are

\[
h_{i,\ell}^{A},
\qquad
h_{i,\ell}^{B^{(1)}},
\qquad
h_{i,\ell}^{B^{(2)}}.
\]

Initialize the A stream by

\[
h_{i,0}^{A}=\alpha(X_i),
\]

where \(\alpha(X_i)=X_i\) may be used when the raw root-frame position is the initial A feature.

Initialize each B stream by

\[
h_{i,0}^{B^{(r)}}=B_i^{(r)}.
\]

A hidden feature satisfies

\[
h_{i,\ell}^{\tau}(g_0\cdot\mathcal C)
=
\rho_\tau(g_0)
h_{i,\ell}^{\tau}(\mathcal C),
\]

where \(\mathcal C\) denotes the complete rooted molecular cloud.

For multiple channels, interpret

\[
h_{i,\ell}^{\tau}
\in
\mathbb R^{c_{\tau,\ell}}\otimes V_\tau.
\]

Only introduce an explicit channel index when necessary:

\[
h_{i,\ell,c}^{\tau}.
\]

Do not place the layer index in a superscript, because superscripts are reserved for representation types.

### 6. Invariant descriptors

Use

\[
s_{ij,\ell}
\]

for the vector of invariant edge descriptors at layer \(\ell\).

Define it generically by

\[
s_{ij,\ell}
=
\operatorname{Inv}_\ell
\left(
d_{ij},
\left\{
h_{i,\ell}^{\tau},
h_{j,\ell}^{\tau}
\right\}_{\tau\in\mathscr T}
\right).
\]

It must satisfy

\[
s_{ij,\ell}(g_0\cdot\mathcal C)
=
s_{ij,\ell}(\mathcal C).
\]

Construct its entries only from complete contractions, invariant inner products, traces, squared norms, or other explicitly verified invariant scalar operations.

Do not use \(I_{ij}\) for invariant descriptors. Reserve

\[
\mathbf 1_n
\]

for the \(n\times n\) identity matrix.

An invariant descriptor does not need to satisfy

\[
s_{ij,\ell}=s_{ji,\ell}
\]

unless the corresponding interaction is explicitly required to be symmetric under exchanging \(i\) and \(j\).

### 7. Covariant maps

Use

\[
C_{\mathrm{out}\leftarrow\mathrm{in}}
\]

for an explicitly specified covariant map when the input and output types should be visible in the notation.

For example,

\[
C_{A\leftarrow A B^{(r)}}:
A\otimes B^{(r)}
\longrightarrow A.
\]

It must satisfy

\[
C_{A\leftarrow A B^{(r)}}
\left(
\rho_A(g)a,
\rho_{B^{(r)}}(g)b
\right)
=
\rho_A(g)
C_{A\leftarrow A B^{(r)}}(a,b).
\]

For higher-order inputs, encode the order in the source label:

\[
C_{A\leftarrow A^p B^{(r)}}:
\operatorname{Sym}^p(A)\otimes B^{(r)}
\longrightarrow A.
\]

The notation \(A^p\) in a map label records input multiplicity only. Actual tensor expressions must still use

\[
a^{\otimes p}.
\]

In general network equations, abbreviate a basis of legal output-\(\tau\) covariants as

\[
C_\alpha^\tau.
\]

Here:

- \(\tau\) is the output representation type;
- \(\alpha\) indexes distinct explicitly constructed covariant maps.

Every \(C_\alpha^\tau\) must be given by a smooth analytic operation such as tensor products, contractions, traces, symmetrization, or traceless projection.

Never construct \(C_\alpha^\tau\) using group averaging.

### 8. Scalar gates

Use

\[
\gamma_{\alpha,\ell}^{\tau}
\]

for a learned scalar gate multiplying \(C_\alpha^\tau\).

It may depend only on invariant descriptors:

\[
\gamma_{\alpha,\ell}^{\tau}
=
\gamma_{\alpha,\ell}^{\tau}(s_{ij,\ell}).
\]

Use smooth scalar functions when differentiability is required. Prefer activations such as SiLU, `tanh`, or `softplus`.

Do not apply an unconstrained componentwise nonlinear function directly to an A-type or B-type tensor.

### 9. Directed messages

Use

\[
m_{j\to i,\ell}^{\tau}
\]

for the type-\(\tau\) message sent from molecule \(j\) to molecule \(i\) at layer \(\ell\).

Define typed messages by

\[
m_{j\to i,\ell}^{\tau}
=
\sum_\alpha
\gamma_{\alpha,\ell}^{\tau}(s_{ij,\ell})
C_\alpha^\tau
\left(
d_{ij},
\left\{
h_{i,\ell}^{\sigma},
h_{j,\ell}^{\sigma}
\right\}_{\sigma\in\mathscr T}
\right).
\]

Every message must satisfy

\[
m_{j\to i,\ell}^{\tau}(g_0\cdot\mathcal C)
=
\rho_\tau(g_0)
m_{j\to i,\ell}^{\tau}(\mathcal C).
\]

Messages are expressed directly in the single root frame. Never introduce a transport operator between nodes.

### 10. Message aggregation

Use

\[
M_{i,\ell}^{\tau}
\]

for the aggregate type-\(\tau\) message received by node \(i\):

\[
M_{i,\ell}^{\tau}
=
\sum_{j\neq i}
m_{j\to i,\ell}^{\tau}.
\]

Use a permutation-symmetric aggregation such as summation.

The aggregation must preserve type:

\[
M_{i,\ell}^{\tau}
\mapsto
\rho_\tau(g_0)M_{i,\ell}^{\tau}.
\]

Do not aggregate A and B features into the same untyped tensor.

### 11. Node updates

Use

\[
\Phi_\ell^\tau
\]

for the type-\(\tau\) node update.

Write

\[
h_{i,\ell+1}^{\tau}
=
\Phi_\ell^\tau
\left(
h_{i,\ell}^{\tau},
M_{i,\ell}^{\tau};
s_{i,\ell}
\right).
\]

The semicolon separates type-\(\tau\) covariants from invariant scalar controls.

The update must satisfy

\[
\Phi_\ell^\tau
\left(
\rho_\tau(g)h,
\rho_\tau(g)M;
s
\right)
=
\rho_\tau(g)
\Phi_\ell^\tau(h,M;s).
\]

Linear maps acting within a type may mix multiplicity channels but must not act arbitrarily on representation coordinates.

### 12. Force output

Use

\[
f_i
\]

for the force on molecule \(i\) expressed in the single root frame.

The force head must have output type A:

\[
f_i
=
\operatorname{Head}_A
\left(
\left\{
h_{i,L}^{\tau}
\right\}_{\tau\in\mathscr T}
\right).
\]

It must satisfy

\[
f_i(g_0\cdot\mathcal C)
=
\rho_A(g_0)f_i(\mathcal C).
\]

Use

\[
F_i
\]

for the physical force in world coordinates:

\[
F_i=O_0f_i.
\]

Never use \(R_i\) to transform \(f_i\) into a molecule-specific local frame.

### 13. Canonical layer equations

Use the following equations as the standard abstract description of one network layer:

\[
s_{ij,\ell}
=
\operatorname{Inv}_\ell
\left(
d_{ij},
h_{i,\ell},
h_{j,\ell}
\right),
\]

\[
m_{j\to i,\ell}^{\tau}
=
\sum_\alpha
\gamma_{\alpha,\ell}^{\tau}(s_{ij,\ell})
C_\alpha^\tau
\left(
d_{ij},
h_{i,\ell},
h_{j,\ell}
\right),
\]

\[
M_{i,\ell}^{\tau}
=
\sum_{j\neq i}
m_{j\to i,\ell}^{\tau},
\]

\[
h_{i,\ell+1}^{\tau}
=
\Phi_\ell^\tau
\left(
h_{i,\ell}^{\tau},
M_{i,\ell}^{\tau};
s_{i,\ell}
\right).
\]

When a formula is written generically in \(h_{i,\ell}\), it denotes the typed collection

\[
h_{i,\ell}
=
\left\{
h_{i,\ell}^{\tau}
\right\}_{\tau\in\mathscr T},
\]

not an untyped concatenation.

### 14. Reserved-symbol summary

| Object | Required notation |
|---|---|
| World-frame position | \(x_i\) |
| World-frame orientation | \(O_i\) |
| Root-frame position | \(X_i\) |
| Root-relative orientation | \(R_i\) |
| Root-frame displacement | \(d_{ij}\) |
| Fixed B template | \(\overline B^{(r)}\) |
| Molecular B descriptor | \(B_i^{(r)}\) |
| Type set | \(\mathscr T\) |
| Representation action | \(\rho_\tau(g)\) |
| Hidden feature | \(h_{i,\ell}^{\tau}\) |
| Invariant descriptor | \(s_{ij,\ell}\) |
| Covariant map | \(C_\alpha^\tau\) |
| Scalar gate | \(\gamma_{\alpha,\ell}^{\tau}\) |
| Directed message | \(m_{j\to i,\ell}^{\tau}\) |
| Aggregated message | \(M_{i,\ell}^{\tau}\) |
| Node update | \(\Phi_\ell^\tau\) |
| Root-frame force | \(f_i\) |
| World-frame force | \(F_i\) |

Do not rename these objects without an explicit project-wide notation change.