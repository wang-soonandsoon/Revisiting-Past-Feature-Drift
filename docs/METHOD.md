# Mathematical core

Features are row vectors. Original edge $W_j$ maps $f_{j-1}$ to $f_j$.
At task $t$, only edges $j<t$ have later observations available for revision.

## Cumulative evidence and prior-centered ridge

For each later task $q>j$, encode the same current image view at the two fixed
endpoints and normalize each feature row to obtain $X_{qj},Y_{qj}$.

$$
G_j=\sum_{q=j+1}^{t}X_{qj}^{\mathsf T}X_{qj},\qquad
H_j=\sum_{q=j+1}^{t}X_{qj}^{\mathsf T}Y_{qj},\qquad
n_j=\sum_{q=j+1}^{t}|\mathcal D_q|.
$$

The original task enters through the original map $W_j$, rather than through
replayed rows in these aggregates. With $\lambda_j=\alpha n_j/d$,

$$
\widehat A_j
=\arg\min_A\sum_{q=j+1}^{t}\|X_{qj}A-Y_{qj}\|_F^2
 +\lambda_j\|A-W_j\|_F^2
=W_j+(G_j+\lambda_j I)^{-1}(H_j-G_jW_j).
$$

The code solves this system through an eigendecomposition and does not form
an explicit matrix inverse. For any unobserved input direction $v$ with
$X_{qj}v=0$ for every contributing task,
$v^{\mathsf T}(\widehat A_j-W_j)=0$. Thus the original map supplies the
unobserved component. The prior-support test checks this property directly.

## Whole-class path selection

For a class born at task $b$, the historical path ends at $t-1$:

$$
C_b=W_{b+1}\cdots W_{t-1},\qquad
D_{b,\alpha}=\widehat A_{b+1,\alpha}\cdots\widehat A_{t-1,\alpha}.
$$

An empty path is the identity. All edges in one candidate birth path share
one $\alpha$. Different birth paths may select different values.

For each held-out current class $k$, subtract only that class's **current**
normalized contributions from each $G_j,H_j,n_j$. Earlier observations remain.
Recompute $\lambda_j$ from the resulting edge count. Score the complete path
using raw held-class means:

$$
J_b(\alpha)=\frac{1}{|\mathcal C_t|}\sum_{k\in\mathcal C_t}
\left\|\left(\bar z_{k,b}D_{b,\alpha}^{(-k)}
-\bar z_{k,t-1}\right)W_t\right\|_2^2.
$$

Classes receive equal weight even when their row counts differ. The native
path is the first candidate and wins numerical ties. After selection, refit
the chosen maps with all allowed observations.

## Means and covariance

For birth statistics $(\mu_{c,b},\Sigma_{c,b})$ and current native statistics
$(\mu^{\mathrm{native}}_{c,t},\Sigma^{\mathrm{native}}_{c,t})$,

$$
\mu^{\mathrm{CHR}}_{c,t}
=\mu^{\mathrm{native}}_{c,t}+\mu_{c,b}(D_b-C_b)W_t.
$$

This corrects the historical contribution and applies the newest bridge
$W_t$ once. Mean CHR keeps $\Sigma^{\mathrm{native}}_{c,t}$.

Suppose the native covariance recursion adds $\nu I$ after each edge. Define
$\Phi_{s:t}=W_{s+1}\cdots W_t$, including $\Phi_{t:t}=I$. Its accumulated
process-noise contribution is

$$
Q_{b,t}=\nu\sum_{s=b+1}^{t}\Phi_{s:t}^{\mathsf T}\Phi_{s:t}.
$$

Joint CHR transports the birth covariance through the selected path while
retaining that original accumulated noise:

$$
\Sigma^{\mathrm{Joint}}_{c,t}
=(D_bW_t)^{\mathsf T}\Sigma_{c,b}(D_bW_t)+Q_{b,t}.
$$

The implementation builds $Q_{b,t}$ backward using native tail products,
without inverting a map. It symmetrizes the result before restoring the
stored covariance precision. Exactly unchanged paths retain the saved state
objects, preserving the original floating-point recursion. Counts and current
class moments are shared across all variants.

## Boundary of this package

The returned statistics are inputs to Gaussian classifier alignment. Alignment
uses a copy of the canonical head, and the canonical learner supplies the next
encoder update. The core package implements historical estimation and moment
revision; the synthetic example demonstrates these operations on CPU. It does
not establish recognition gains on a real dataset.
