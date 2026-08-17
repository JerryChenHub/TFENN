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

Let \(G\) be the symmetry group and let \(\mathcal T\) be the registered set of representation types.

For each \(\sigma\in\mathcal T\):

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

The target type is \(\tau\in\mathcal T\), with output space \(V_\tau\), dimension \(d_\tau\), and action \(\rho_\tau(g)\).

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
| \(R\) | Latent channel-mixing rank |
| \(q\in\{1,\ldots,R\}\) | Latent channel index |
| \(P^{(t)}\in\mathbb R^{R\times c_t}\) | Trainable input-channel projection for slot \(t\) |
| \(O\in\mathbb R^{c_o\times R}\) | Trainable output-channel projection |

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
y_{c_o}
=
\sum_{q=1}^{R}O_{c_o,q}
\sum_{\alpha=1}^{n_{\tau\leftarrow\Sigma}}
\gamma^{\tau}_{\Sigma,q,\alpha}(\mathbf{s})
v_{q,\alpha}.
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
| \(P^{(t)}\), \(O\) | Trainable | Mix input and output channels |
| Gate encoder/head parameters | Trainable | Produce invariant coefficients |
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

