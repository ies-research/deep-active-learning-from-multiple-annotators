# Local Pi-Mixture Response Channel with Response-Collapse Regularization

## Goal

Implement a local annotator-response model for a candidate instance--annotator pair `(n, m)` that preserves the original correctness/fallback channel, but augments it with an explicit class-independent response component.

The motivation is:

- the original channel can work well for normal annotators;
- deterministic or low-entropy spammers can still be overvalued;
- a class-independent response component can represent local response collapse directly;
- an optional `H(g_nm)` cap acts as a robust final safeguard.

The local mixture channel is

```latex
K_{mix,nm}(l \mid c)
=
(1-\pi_{nm})K_{orig,nm}(l \mid c)
+
\pi_{nm}g_{nm}^{(l)}.
```

Here:

- `c` is the latent true class.
- `l` is the label emitted by annotator `m`.
- `K_orig,nm(l | c)` is the original correctness/fallback response channel.
- `g_nm` is the local marginal response distribution of annotator `m` around candidate instance `x_n`.
- `pi_nm` is the local probability of class-independent response behavior.

If `pi_nm = 0`, the model reduces to the original channel.

If `pi_nm = 1`, the response becomes class-independent:

```latex
K_{mix,nm}(l \mid c)=g_{nm}^{(l)} \quad \forall c,
```

so the annotator response carries no information about the latent class.

---

## 1. Inputs

For a candidate pair `(n, m)`, assume the following inputs.

### Data and dimensions

- `C`: number of classes.
- `candidate_idx = n`.
- `annotator_idx = m`.
- `observed_indices_m`: indices `i` previously labeled by annotator `m`.
- `z_im[i]`: observed label from annotator `m` on instance `i`, integer in `{0, ..., C-1}`.
- `w_ni[i]`: kernel similarity between candidate instance `x_n` and observed instance `x_i`.

### Classifier beliefs

For the candidate instance:

- `p_n`: class-probability vector `p_n in Delta_C`, or samples from a posterior over `p_n`.

For previous instances:

- `p_i`: class-probability vectors `p_i in Delta_C`, used only if estimating `mu_nm` or `pi_nm` with likelihood-based diagnostics. The simple entropy-based `pi_nm` estimator only needs observed labels and kernel weights.

### Original channel parameters

The original channel uses:

- `mu_nm in [0, 1]`: local correctness / accuracy parameter.
- `g_nm in Delta_C`: local fallback / marginal response distribution.

If `mu_nm` is uncertain, sample it from the current posterior, e.g.

```latex
\mu_{nm}^{(s)} \sim p(\mu_{nm}\mid \mathcal D_m,x_n).
```

If the current implementation already has a Beta posterior over `mu_nm`, keep using it.

### Hyperparameters

- `beta0`: Dirichlet prior pseudo-count for `g_nm`, e.g. `0.5` or `1.0`.
- `kappa_pi`: ESS shrinkage constant for `pi_nm`, e.g. `3`, `5`, or `10`.
- `gamma_pi`: optional exponent for response-collapse score, e.g. `1.0` or `2.0`.
- `pi_max`: optional maximum value for `pi_nm`, e.g. `1.0`.
- `lambda_g`: multiplier for the optional `H(g_nm)` cap, e.g. `1.0`.
- `eps`: small numerical constant, e.g. `1e-12`.

---

## 2. Estimate the local marginal response distribution `g_nm`

For candidate pair `(n, m)`, estimate `g_nm` from kernel-weighted label counts of annotator `m`:

```latex
\beta_{nm}^{(l)}
=
\beta_0
+
\sum_{i\in\mathcal D_m}
w_{ni}\mathbb 1[z_{im}=l].
```

Then

```latex
\boldsymbol g_{nm}\sim \operatorname{Dirichlet}(\boldsymbol\beta_{nm}).
```

The posterior mean is

```latex
\bar g_{nm}^{(l)}
=
\frac{\beta_{nm}^{(l)}}{\sum_{u=1}^C \beta_{nm}^{(u)}}.
```

Implementation:

```python
import numpy as np


def local_g_posterior(
    C: int,
    observed_indices_m,
    z_im,
    w_ni,
    beta0: float = 0.5,
):
    beta = beta0 * np.ones(C, dtype=float)
    for i in observed_indices_m:
        beta[int(z_im[i])] += float(w_ni[i])
    g_mean = beta / beta.sum()
    return beta, g_mean
```

---

## 3. Define the original correctness/fallback channel

The original channel is

```latex
K_{orig,nm}(l \mid c)
=
\begin{cases}
\mu_{nm}, & l=c,\\[4pt]
(1-\mu_{nm})
\dfrac{g_{nm}^{(l)}}{1-g_{nm}^{(c)}}, & l\neq c.
\end{cases}
```

This means:

- with probability `mu_nm`, the annotator emits the true class;
- otherwise, the annotator emits a wrong label according to `g_nm` after removing the true class and renormalizing.

Implementation note: because the denominator `1 - g_nm[c]` can become very small if `g_nm` is nearly deterministic, use smoothing and numerical guards. After building each row, renormalize the row to sum to one.

```python
def build_original_channel(mu: float, g: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Build K_orig[l | c] as matrix K[c, l].

    Rows correspond to true class c.
    Columns correspond to emitted label l.
    """
    C = len(g)
    g = np.asarray(g, dtype=float)
    g = np.clip(g, eps, 1.0)
    g = g / g.sum()

    mu = float(np.clip(mu, 0.0, 1.0))

    K = np.zeros((C, C), dtype=float)

    for c in range(C):
        denom = max(1.0 - g[c], eps)

        for l in range(C):
            if l == c:
                K[c, l] = mu
            else:
                K[c, l] = (1.0 - mu) * g[l] / denom

        # Numerical safety. The row should already sum to one if denom is fine.
        row_sum = K[c].sum()
        if row_sum <= eps:
            K[c] = np.ones(C) / C
        else:
            K[c] /= row_sum

    return K
```

---

## 4. Estimate the local mixture weight `pi_nm`

The simplest useful estimate is based on local response collapse.

Define the entropy of the local marginal response distribution:

```latex
H(\boldsymbol g_{nm})
=
-\sum_{l=1}^C g_{nm}^{(l)}\log g_{nm}^{(l)}.
```

Normalize it by `log(C)`:

```latex
\tilde H(\boldsymbol g_{nm})
=
\frac{H(\boldsymbol g_{nm})}{\log C}.
```

Define response collapse as

```latex
\mathrm{collapse}_{nm}
=
1-\tilde H(\boldsymbol g_{nm}).
```

Thus:

- if `g_nm` is uniform, `collapse_nm approx 0`;
- if `g_nm` is deterministic, `collapse_nm approx 1`.

Now compute the local effective sample size from kernel weights:

```latex
N_{nm}^{eff}
=
\frac{
\left(\sum_{i\in\mathcal D_m}w_{ni}\right)^2
}{
\sum_{i\in\mathcal D_m}w_{ni}^2
}.
```

Shrink the collapse score when local evidence is weak:

```latex
\lambda_{nm}
=
\frac{
N_{nm}^{eff}
}{
N_{nm}^{eff}+\kappa_\pi
}.
```

Then define

```latex
\pi_{nm}
=
\min\left\{
\pi_{max},
\lambda_{nm}
\left(1-\frac{H(\boldsymbol g_{nm})}{\log C}\right)^{\gamma_\pi}
\right\}.
```

The exponent `gamma_pi` controls aggressiveness:

- `gamma_pi = 1`: linear response-collapse score;
- `gamma_pi > 1`: only very concentrated `g_nm` creates high `pi_nm`.

Implementation:

```python
def entropy(prob: np.ndarray, eps: float = 1e-12) -> float:
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, eps, 1.0)
    prob = prob / prob.sum()
    return float(-np.sum(prob * np.log(prob)))


def effective_sample_size(weights, eps: float = 1e-12) -> float:
    weights = np.asarray(weights, dtype=float)
    s1 = weights.sum()
    s2 = np.sum(weights ** 2)
    if s1 <= eps or s2 <= eps:
        return 0.0
    return float((s1 ** 2) / (s2 + eps))


def estimate_pi_from_response_collapse(
    g: np.ndarray,
    local_weights,
    kappa_pi: float = 5.0,
    gamma_pi: float = 1.0,
    pi_max: float = 1.0,
    eps: float = 1e-12,
) -> float:
    C = len(g)
    H_g = entropy(g, eps=eps)
    H_max = np.log(C)

    if H_max <= eps:
        return 0.0

    collapse = 1.0 - H_g / H_max
    collapse = float(np.clip(collapse, 0.0, 1.0))

    N_eff = effective_sample_size(local_weights, eps=eps)
    lambda_eff = N_eff / (N_eff + kappa_pi + eps)

    pi = lambda_eff * (collapse ** gamma_pi)
    pi = min(float(pi), float(pi_max))
    pi = float(np.clip(pi, 0.0, 1.0))
    return pi
```

---

## 5. Build the pi-mixture channel

The local mixture channel is

```latex
K_{mix,nm}(l \mid c)
=
(1-\pi_{nm})K_{orig,nm}(l \mid c)
+
\pi_{nm}g_{nm}^{(l)}.
```

In matrix form, for every true class `c`, the class-independent component has the same row `g_nm`.

Implementation:

```python
def build_pi_mixture_channel(
    mu: float,
    g: np.ndarray,
    pi: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build K_mix[c, l] = (1 - pi) K_orig[c, l] + pi g[l].
    """
    K_orig = build_original_channel(mu=mu, g=g, eps=eps)

    g = np.asarray(g, dtype=float)
    g = np.clip(g, eps, 1.0)
    g = g / g.sum()

    pi = float(np.clip(pi, 0.0, 1.0))

    K_mix = (1.0 - pi) * K_orig + pi * g[None, :]

    # Numerical safety.
    K_mix = np.clip(K_mix, eps, 1.0)
    K_mix = K_mix / K_mix.sum(axis=1, keepdims=True)

    return K_mix
```

---

## 6. Compute predictive queried-label distribution

For candidate class belief `p_n`, compute

```latex
q_{nm}^{(l)}
=
\Pr(Z_{nm}=l\mid \boldsymbol p_n,K_{mix,nm})
=
\sum_{c=1}^C p_n^{(c)}K_{mix,nm}(l\mid c).
```

Implementation:

```python
def predictive_label_distribution(p: np.ndarray, K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    p: shape [C]
    K: shape [C, C], K[c, l]
    returns q: shape [C], q[l]
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()

    q = p @ K
    q = np.clip(q, eps, 1.0)
    q = q / q.sum()
    return q
```

---

## 7. Compute posterior class probabilities after hypothetical response

After hypothetically observing label `l`, update the class belief via Bayes' rule:

```latex
\tilde p_{nm,l}^{(c)}
=
\frac{
p_n^{(c)}K_{mix,nm}(l\mid c)
}{
q_{nm}^{(l)}
}.
```

Implementation:

```python
def posterior_after_label(
    p: np.ndarray,
    K: np.ndarray,
    q: np.ndarray,
    l: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Posterior over true classes after observing emitted label l.
    """
    numerator = p * K[:, l]
    denom = max(float(q[l]), eps)

    post = numerator / denom
    post = np.clip(post, eps, 1.0)
    post = post / post.sum()
    return post
```

---

## 8. Compute expected KL information gain

For KL-based gain,

```latex
G(\boldsymbol p,\tilde{\boldsymbol p})
=
KL(\tilde{\boldsymbol p}\|\boldsymbol p),
```

the expected gain is

```latex
G_{nm}
=
\sum_{l=1}^C
q_{nm}^{(l)}
KL
\left(
\tilde{\boldsymbol p}_{nm,l}
\| 
\boldsymbol p_n
\right).
```

This is mutual information under the chosen channel.

Implementation:

```python
def kl_div(posterior: np.ndarray, prior: np.ndarray, eps: float = 1e-12) -> float:
    posterior = np.asarray(posterior, dtype=float)
    prior = np.asarray(prior, dtype=float)

    posterior = np.clip(posterior, eps, 1.0)
    posterior = posterior / posterior.sum()

    prior = np.clip(prior, eps, 1.0)
    prior = prior / prior.sum()

    return float(np.sum(posterior * (np.log(posterior) - np.log(prior))))


def expected_kl_gain(p: np.ndarray, K: np.ndarray, eps: float = 1e-12) -> float:
    q = predictive_label_distribution(p, K, eps=eps)

    gain = 0.0
    for l in range(len(p)):
        post_l = posterior_after_label(p, K, q, l, eps=eps)
        gain += q[l] * kl_div(post_l, p, eps=eps)

    return float(gain)
```

---

## 9. Optional caps and regularizers

### 9.1 Mutual-information feasibility cap

Exact mutual information satisfies

```latex
0 \leq I(Y;Z) \leq \min\{H(\boldsymbol p_n),H(\boldsymbol q_{nm})\}.
```

If the gain is approximated, project it onto this feasible interval:

```latex
G_{nm}
\leftarrow
\min
\left\{
\max(0,G_{nm}),
\min\left[
H(\boldsymbol p_n),
H(\boldsymbol q_{nm})
\right]
\right\}.
```

### 9.2 Response-collapse cap

Empirically, it can be useful to also cap by the entropy of the local marginal response distribution:

```latex
G_{nm}
\leftarrow
\min
\left\{
G_{nm},
\lambda_g H(\boldsymbol g_{nm})
\right\}.
```

Important interpretation:

```latex
H(\boldsymbol g_{nm})
```

is **not** an information-theoretic upper bound on mutual information. It is a response-collapse regularizer. It suppresses annotators whose local marginal response distribution is nearly deterministic.

Combined cap:

```latex
G_{nm}^{rob}
=
\min
\left\{
\max(0,G_{nm}),
H(\boldsymbol p_n),
H(\boldsymbol q_{nm}),
\lambda_g H(\boldsymbol g_{nm})
\right\}.
```

Implementation:

```python
def cap_gain(
    gain: float,
    p: np.ndarray,
    q: np.ndarray,
    g: np.ndarray | None = None,
    use_mi_cap: bool = True,
    use_g_cap: bool = True,
    lambda_g: float = 1.0,
    eps: float = 1e-12,
) -> float:
    gain = max(float(gain), 0.0)

    caps = []

    if use_mi_cap:
        caps.append(entropy(p, eps=eps))
        caps.append(entropy(q, eps=eps))

    if use_g_cap and g is not None:
        caps.append(lambda_g * entropy(g, eps=eps))

    if len(caps) > 0:
        gain = min(gain, min(caps))

    return float(gain)
```

---

## 10. Monte Carlo scoring with empirical UCB

If `mu_nm`, `g_nm`, and/or `p_n` are uncertain, compute a sampled gain distribution.

For `s = 1, ..., S`:

1. Sample or set candidate class belief `p_n^(s)`.
2. Sample local response distribution:

```latex
\boldsymbol g_{nm}^{(s)}\sim \operatorname{Dirichlet}(\boldsymbol\beta_{nm}).
```

3. Sample or set original-channel accuracy:

```latex
\mu_{nm}^{(s)}.
```

4. Estimate `pi_nm^(s)` from `g_nm^(s)` and local weights.
5. Build `K_mix,nm^(s)`.
6. Compute gain and optional caps.

Then compute empirical UCB:

```latex
UCB_{nm}
=
\bar G_{nm}
+
\lambda_{ucb}\hat\sigma_{nm}.
```

Implementation skeleton:

```python
def score_candidate_pair_pi_mixture(
    p_n_mean: np.ndarray,
    C: int,
    observed_indices_m,
    z_im,
    w_ni,
    beta0: float,
    mu_sampler,
    p_n_sampler=None,
    S: int = 10,
    kappa_pi: float = 5.0,
    gamma_pi: float = 1.0,
    pi_max: float = 1.0,
    lambda_g: float = 1.0,
    lambda_ucb: float = 0.0,
    use_mi_cap: bool = True,
    use_g_cap: bool = True,
    eps: float = 1e-12,
):
    """
    Parameters
    ----------
    p_n_mean:
        Deterministic candidate class belief. Used if p_n_sampler is None.
    mu_sampler:
        Callable returning one sample of mu_nm in [0, 1].
        If mu is deterministic, pass: lambda: mu_value.
    p_n_sampler:
        Optional callable returning one sample of p_n.
        If None, use p_n_mean for all samples.
    """

    beta_nm, g_mean = local_g_posterior(
        C=C,
        observed_indices_m=observed_indices_m,
        z_im=z_im,
        w_ni=w_ni,
        beta0=beta0,
    )

    local_weights = np.asarray([w_ni[i] for i in observed_indices_m], dtype=float)

    gains = []
    pi_samples = []

    for _ in range(S):
        # 1. sample p_n
        if p_n_sampler is None:
            p_s = np.asarray(p_n_mean, dtype=float)
            p_s = np.clip(p_s, eps, 1.0)
            p_s = p_s / p_s.sum()
        else:
            p_s = p_n_sampler()
            p_s = np.asarray(p_s, dtype=float)
            p_s = np.clip(p_s, eps, 1.0)
            p_s = p_s / p_s.sum()

        # 2. sample g_nm
        g_s = np.random.dirichlet(beta_nm)

        # 3. sample mu_nm
        mu_s = float(np.clip(mu_sampler(), 0.0, 1.0))

        # 4. estimate pi_nm
        pi_s = estimate_pi_from_response_collapse(
            g=g_s,
            local_weights=local_weights,
            kappa_pi=kappa_pi,
            gamma_pi=gamma_pi,
            pi_max=pi_max,
            eps=eps,
        )
        pi_samples.append(pi_s)

        # 5. build channel
        K_s = build_pi_mixture_channel(
            mu=mu_s,
            g=g_s,
            pi=pi_s,
            eps=eps,
        )

        # 6. compute gain
        q_s = predictive_label_distribution(p_s, K_s, eps=eps)
        gain_s = expected_kl_gain(p_s, K_s, eps=eps)

        # 7. optional caps
        gain_s = cap_gain(
            gain=gain_s,
            p=p_s,
            q=q_s,
            g=g_s,
            use_mi_cap=use_mi_cap,
            use_g_cap=use_g_cap,
            lambda_g=lambda_g,
            eps=eps,
        )

        gains.append(gain_s)

    gains = np.asarray(gains, dtype=float)
    pi_samples = np.asarray(pi_samples, dtype=float)

    mean_gain = float(gains.mean())
    std_gain = float(gains.std(ddof=1)) if len(gains) > 1 else 0.0
    score = mean_gain + lambda_ucb * std_gain

    return {
        "score": score,
        "mean_gain": mean_gain,
        "std_gain": std_gain,
        "gains": gains,
        "pi_mean": float(pi_samples.mean()),
        "pi_samples": pi_samples,
        "beta_nm": beta_nm,
        "g_mean": g_mean,
    }
```

---

## 11. Optional likelihood-based pi estimator

The entropy-based `pi_nm` estimator is the lightweight default. A more model-based alternative compares the original channel with the class-independent response channel.

For each previous observed label `z_im`, define

```latex
L_i^{orig}
=
\sum_{c=1}^C p_i^{(c)}K_{orig,nm}(z_{im}\mid c),
```

and

```latex
L_i^{ind}
=
g_{nm}^{(z_{im})}.
```

Then define posterior log-odds:

```latex
\log\frac{\pi_{nm}}{1-\pi_{nm}}
=
\log\frac{\pi_0}{1-\pi_0}
+
\gamma_{ess}
\sum_{i\in\mathcal D_m}
w_{ni}
\log
\frac{
L_i^{ind}
}{
L_i^{orig}
}.
```

where

```latex
\gamma_{ess}
=
\frac{
N_{nm}^{eff}
}{
N_{nm}^{eff}+\kappa_\pi
}.
```

Then

```latex
\pi_{nm}
=
\sigma
\left(
\log\frac{\pi_{nm}}{1-\pi_{nm}}
\right).
```

This version uses the labels and classifier beliefs more directly, but it may be more sensitive to classifier confirmation bias. The entropy-based version is simpler and often more robust for deterministic response collapse.

---

## 12. Recommended ablations

Evaluate at least the following variants.

### A. Original channel

```latex
K_{orig}.
```

### B. Original channel + `H(g)` cap

```latex
G^{rob}
=
\min\{G^{orig},\lambda_g H(\boldsymbol g)\}.
```

This is the strong empirical baseline if it already works well.

### C. Pi-mixture without `H(g)` cap

```latex
K_{mix}
=
(1-\pi)K_{orig}+\pi g.
```

This tests whether the explicit class-independent channel is enough.

### D. Pi-mixture + `H(g)` cap

```latex
G^{rob}
=
\min\{G^{mix},\lambda_g H(\boldsymbol g)\}.
```

This is likely the most robust variant.

---

## 13. Important notes

1. Use the same channel `K_mix` for:
   - predictive label distribution `q`,
   - posterior update `p_tilde`,
   - information gain.

2. The `H(q)` cap is an information-theoretic feasibility cap.

3. The `H(g)` cap is **not** a mutual-information bound. It is a response-collapse regularizer.

4. The entropy-based `pi_nm` estimator is deliberately simple. It turns local low-entropy response behavior into a higher probability of class-independent emission.

5. If the `H(g)` cap already works well, do not remove it immediately. Test the pi-mixture both with and without the cap.

6. Be careful with the original fallback channel when `g[c] approx 1`. Use smoothing, numerical guards, and row renormalization.

7. If local neighborhoods are truly class-homogeneous, `H(g)` may suppress good annotators. The pi-mixture may help in these cases because it softens the penalty by mixing rather than hard-capping.
