# Likelihood-Based Local Task-Dependence Estimation for the Response-Bias Mixture Channel

## Goal

For each candidate instance--annotator pair \((n,m)\), estimate a **local annotator-response channel**

\[
K_{nm}(l \mid c)
=
\theta_{nm}\mathbb{1}[l=c]
+
(1-\theta_{nm})g_{nm}^{(l)},
\]

where:

- \(c\in\{1,\dots,C\}\) is the latent true class.
- \(l\in\{1,\dots,C\}\) is the label emitted by annotator \(m\).
- \(\boldsymbol{g}_{nm}\in\Delta_C\) is the **local marginal response distribution** of annotator \(m\) around candidate instance \(x_n\).
- \(\theta_{nm}\in[0,1]\) is the **local task-dependence strength** of annotator \(m\) around candidate instance \(x_n\).

Interpretation:

- With probability \(\theta_{nm}\), the annotator emits the true class.
- With probability \(1-\theta_{nm}\), the annotator emits a label according to their local response bias \(\boldsymbol{g}_{nm}\).

This model separates **local response bias** from **local task-dependent information**. A deterministic local spammer can be represented by \(\theta_{nm}\approx 0\) and a concentrated \(\boldsymbol{g}_{nm}\), which makes the response nearly class-independent.

---

## Why use the likelihood solution?

The proxy estimator first computes pseudo-values

\[
\hat\theta_{im\mid n}
=
\operatorname{clip}
\left(
\frac{h_i-b_{ni}}{r_i-b_{ni}},0,1
\right),
\]

and then smooths these pseudo-observations. This can fail because it involves unstable ratios, clipping, and artificial per-observation \(\theta\)-targets.

The likelihood solution avoids this. It estimates the candidate-local parameter \(\theta_{nm}\) directly from the response channel. For every already observed label \(z_{im}\) from annotator \(m\), the predictive probability under the channel is

\[
\Pr(z_{im}\mid \boldsymbol{p}_i,\theta_{nm},\boldsymbol{g}_{nm})
=
\theta_{nm}p_i^{(z_{im})}
+
(1-\theta_{nm})g_{nm}^{(z_{im})}.
\]

This likelihood compares two explanations:

- \(p_i^{(z_{im})}\): the observed label is likely because it matches the classifier belief.
- \(g_{nm}^{(z_{im})}\): the observed label is likely because the annotator tends to emit this label locally anyway.

If observed labels are better explained by classifier beliefs than by local response bias, the posterior over \(\theta_{nm}\) shifts upward. If they are better explained by local response bias, the posterior shifts downward.

---

## Inputs

Assume the implementation has:

- `C`: number of classes.
- `candidate_idx = n`.
- `annotator_idx = m`.
- `observed_indices_m`: indices \(i\) already labeled by annotator \(m\).
- `z_im[i]`: observed label from annotator \(m\) for instance \(i\), integer in `{0, ..., C-1}`.
- `w_ni[i]`: kernel similarity between candidate instance \(x_n\) and observed instance \(x_i\).
- `p_mean[i, c]`: posterior mean classifier belief \(\bar p_i^{(c)}\), or deterministic classifier probability.
- Optional `alpha_p[i, c]`: Dirichlet parameters for classifier belief if sampling \(\boldsymbol{p}_i\) is desired.
- `beta0`: Dirichlet prior pseudo-count for the local response-bias distribution.
- `a_theta, b_theta`: Beta prior parameters for \(\theta_{nm}\).
- `theta_grid`: grid points in \((0,1)\), e.g. 64 points.
- `eps`: small numerical constant, e.g. `1e-12`.

---

# Step 1: Estimate the local response-bias posterior \(\boldsymbol{g}_{nm}\)

Use kernel-weighted label counts from annotator \(m\):

\[
\beta_{nm}^{(l)}
=
\beta_0
+
\sum_{i\in\mathcal{D}_m}
w_{ni}\mathbb{1}[z_{im}=l].
\]

Then

\[
\boldsymbol{g}_{nm}\sim \operatorname{Dir}(\boldsymbol{\beta}_{nm}).
\]

Implementation:

```python
import numpy as np


def local_response_bias_posterior(
    observed_indices_m,
    z_im,
    w_ni,
    C,
    beta0,
):
    beta_nm = beta0 * np.ones(C, dtype=float)
    for i in observed_indices_m:
        label = int(z_im[i])
        beta_nm[label] += float(w_ni[i])
    return beta_nm
```

The posterior mean is

\[
\bar g_{nm}^{(l)}
=
\frac{\beta_{nm}^{(l)}}{\sum_u \beta_{nm}^{(u)}}.
\]

---

# Step 2: Define the local likelihood for \(\theta_{nm}\)

For fixed \(\boldsymbol{g}_{nm}\) and classifier beliefs \(\boldsymbol{p}_i\), the likelihood contribution of an observed label \(z_{im}\) is

\[
L_i(\theta_{nm})
=
\theta_{nm}p_i^{(z_{im})}
+
(1-\theta_{nm})g_{nm}^{(z_{im})}.
\]

The local posterior over \(\theta_{nm}\) is

\[
p(\theta_{nm}\mid \mathcal{D}_m,\boldsymbol{g}_{nm})
\propto
\theta_{nm}^{a_\theta-1}
(1-\theta_{nm})^{b_\theta-1}
\prod_{i\in\mathcal{D}_m}
\left[
\theta_{nm}p_i^{(z_{im})}
+
(1-\theta_{nm})g_{nm}^{(z_{im})}
\right]^{w_{ni}}.
\]

The log-posterior on a grid \(t_r\in(0,1)\) is

\[
\log \omega_r
=
(a_\theta-1)\log t_r
+
(b_\theta-1)\log(1-t_r)
+
\sum_{i\in\mathcal{D}_m}
w_{ni}
\log
\left[
t_r p_i^{(z_{im})}
+
(1-t_r)g_{nm}^{(z_{im})}
\right].
\]

Implementation:

```python

def theta_log_posterior_grid(
    theta_grid,
    observed_indices_m,
    z_im,
    w_ni,
    p_mean,
    g_vec,
    a_theta,
    b_theta,
    eps=1e-12,
):
    """Compute unnormalized log posterior over theta on a fixed grid.

    Parameters
    ----------
    theta_grid : np.ndarray, shape [R]
        Grid values in (0, 1).
    observed_indices_m : iterable
        Instances already labeled by annotator m.
    z_im : dict or array-like
        Observed labels from annotator m.
    w_ni : dict or array-like
        Kernel weights between candidate n and observed instance i.
    p_mean : np.ndarray, shape [N, C]
        Classifier probability vectors or posterior means.
    g_vec : np.ndarray, shape [C]
        Sampled or mean local response-bias distribution.
    a_theta, b_theta : float
        Beta prior parameters.
    eps : float
        Numerical stability constant.

    Returns
    -------
    log_post : np.ndarray, shape [R]
        Unnormalized log posterior values.
    """
    theta_grid = np.asarray(theta_grid, dtype=float)
    theta_grid = np.clip(theta_grid, eps, 1.0 - eps)

    log_post = (
        (a_theta - 1.0) * np.log(theta_grid)
        + (b_theta - 1.0) * np.log1p(-theta_grid)
    )

    for i in observed_indices_m:
        label = int(z_im[i])
        w = float(w_ni[i])
        if w <= 0.0:
            continue

        p_label = float(p_mean[i, label])
        g_label = float(g_vec[label])

        likelihood = theta_grid * p_label + (1.0 - theta_grid) * g_label
        log_post += w * np.log(np.clip(likelihood, eps, 1.0))

    return log_post
```

---

# Step 3: Normalize the grid posterior and sample \(\theta_{nm}\)

Normalize stably with log-sum-exp:

\[
\omega_r
=
\frac{\exp(\log\omega_r)}{\sum_u \exp(\log\omega_u)}.
\]

Implementation:

```python

def normalize_log_weights(log_w, eps=1e-12):
    log_w = np.asarray(log_w, dtype=float)
    max_log_w = np.max(log_w)
    weights = np.exp(log_w - max_log_w)
    total = weights.sum()
    if total <= eps or not np.isfinite(total):
        return np.ones_like(weights) / len(weights)
    return weights / total


def sample_theta_from_grid(theta_grid, probs, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.choice(len(theta_grid), p=probs)
    return float(theta_grid[idx])
```

Posterior summary:

```python

def theta_grid_summary(theta_grid, probs):
    mean = float(np.sum(theta_grid * probs))
    var = float(np.sum(((theta_grid - mean) ** 2) * probs))
    mode = float(theta_grid[np.argmax(probs)])
    return mean, var, mode
```

---

# Step 4: Recommended Monte Carlo sampling scheme

For each acquisition sample \(s=1,\dots,S\):

1. Sample local response bias:

\[
\boldsymbol{g}_{nm}^{(s)}\sim \operatorname{Dir}(\boldsymbol{\beta}_{nm}).
\]

2. Compute the grid posterior

\[
p(\theta_{nm}\mid\mathcal{D}_m,\boldsymbol{g}_{nm}^{(s)})
\]

using the likelihood above.

3. Sample

\[
\theta_{nm}^{(s)}
\]

from the normalized grid posterior.

4. Sample or plug in candidate class belief:

\[
\boldsymbol{p}_{n}^{(s)}.
\]

5. Compute gain under

\[
K_{nm}^{(s)}(l\mid c)
=
\theta_{nm}^{(s)}\mathbb{1}[l=c]
+
(1-\theta_{nm}^{(s)})g_{nm}^{(s,l)}.
\]

Implementation:

```python

def sample_local_channel_parameters(
    beta_nm,
    theta_grid,
    observed_indices_m,
    z_im,
    w_ni,
    p_mean,
    a_theta,
    b_theta,
    rng=None,
    eps=1e-12,
):
    """Sample g_nm and theta_nm for one candidate pair (n,m).

    Returns
    -------
    theta_s : float
        Sampled local task-dependence strength.
    g_s : np.ndarray, shape [C]
        Sampled local response-bias distribution.
    theta_probs : np.ndarray, shape [R]
        Normalized grid posterior over theta conditional on sampled g_s.
    """
    if rng is None:
        rng = np.random.default_rng()

    g_s = rng.dirichlet(beta_nm)

    log_post = theta_log_posterior_grid(
        theta_grid=theta_grid,
        observed_indices_m=observed_indices_m,
        z_im=z_im,
        w_ni=w_ni,
        p_mean=p_mean,
        g_vec=g_s,
        a_theta=a_theta,
        b_theta=b_theta,
        eps=eps,
    )
    theta_probs = normalize_log_weights(log_post, eps=eps)
    theta_s = sample_theta_from_grid(theta_grid, theta_probs, rng=rng)

    return theta_s, g_s, theta_probs
```

---

# Step 5: Compute predictive queried-label distribution

Given candidate belief \(\boldsymbol{p}_n\), sampled \(\theta\), and sampled \(\boldsymbol{g}\), the predictive queried-label distribution is

\[
q_{nm}^{(l)}
=
\theta_{nm}p_n^{(l)}
+
(1-\theta_{nm})g_{nm}^{(l)}.
\]

Implementation:

```python

def predictive_label_distribution(p_n, theta, g, eps=1e-12):
    q = theta * p_n + (1.0 - theta) * g
    q = np.clip(q, eps, 1.0)
    q = q / q.sum()
    return q
```

---

# Step 6: Compute posterior class beliefs after hypothetical labels

After observing hypothetical label \(l\), Bayes' rule gives

\[
\tilde p_l^{(c)}
=
\frac{
p_n^{(c)}
\left[
\theta\mathbb{1}[l=c]
+
(1-\theta)g^{(l)}
\right]
}{
q^{(l)}
}.
\]

Implementation:

```python

def posterior_after_label(p_n, theta, g, label, eps=1e-12):
    C = len(p_n)
    likelihood = (1.0 - theta) * g[label] * np.ones(C, dtype=float)
    likelihood[label] += theta

    unnorm = p_n * likelihood
    denom = unnorm.sum()
    if denom <= eps or not np.isfinite(denom):
        return p_n.copy()
    return unnorm / denom
```

---

# Step 7: Compute sampled KL information gain

For KL-based gain,

\[
G^{(s)}_{nm}
=
\sum_{l=1}^C
q_{nm}^{(s,l)}
\operatorname{KL}
\left(
\tilde{\boldsymbol p}_{l}^{(s)}
\Vert
\boldsymbol p_n^{(s)}
\right).
\]

Implementation:

```python

def entropy(prob, eps=1e-12):
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, eps, 1.0)
    prob = prob / prob.sum()
    return float(-np.sum(prob * np.log(prob)))


def kl_divergence(posterior, prior, eps=1e-12):
    posterior = np.asarray(posterior, dtype=float)
    prior = np.asarray(prior, dtype=float)
    posterior = np.clip(posterior, eps, 1.0)
    prior = np.clip(prior, eps, 1.0)
    posterior = posterior / posterior.sum()
    prior = prior / prior.sum()
    return float(np.sum(posterior * (np.log(posterior) - np.log(prior))))


def sampled_information_gain(p_n, theta, g, eps=1e-12):
    C = len(p_n)
    q = predictive_label_distribution(p_n, theta, g, eps=eps)

    gain = 0.0
    for label in range(C):
        post = posterior_after_label(p_n, theta, g, label, eps=eps)
        gain += q[label] * kl_divergence(post, p_n, eps=eps)

    return float(gain), q
```

---

# Step 8: Optional caps and regularizers

## 8.1 Mutual-information feasibility cap

Exact mutual information satisfies

\[
0\leq I(Y;Z)\leq \min\{H(\boldsymbol p_n),H(\boldsymbol q_{nm})\}.
\]

For approximate gains, project the sampled gain into this feasible interval:

\[
G^{(s)}
\leftarrow
\min
\left\{
\max\{0,G^{(s)}\},
\min[H(\boldsymbol p_n^{(s)}),H(\boldsymbol q_{nm}^{(s)})]
\right\}.
\]

This is an information-theoretic correction.

## 8.2 Response-collapse cap using \(H(\boldsymbol g_{nm})\)

Empirically, deterministic or highly concentrated local response distributions can still cause high scores if \(\theta\) is overestimated. A practical robustification is to also cap the gain by the entropy of the local response-bias distribution:

\[
G^{(s)}
\leftarrow
\min
\left\{
G^{(s)},
\lambda_g H(\boldsymbol g_{nm}^{(s)})
\right\}.
\]

Important: \(H(\boldsymbol g_{nm})\) is **not** a mutual-information upper bound. It is a response-collapse regularizer. It suppresses annotators whose local marginal responses are nearly deterministic.

A combined robust cap is

\[
G^{(s)}_{\mathrm{rob}}
=
\min
\left\{
\max\{0,G^{(s)}\},
H(\boldsymbol p_n^{(s)}),
H(\boldsymbol q_{nm}^{(s)}),
\lambda_g H(\boldsymbol g_{nm}^{(s)})
\right\}.
\]

Implementation:

```python

def robust_cap_gain(
    gain,
    p_n,
    q,
    g,
    use_mi_cap=True,
    use_g_entropy_cap=True,
    lambda_g=1.0,
    eps=1e-12,
):
    gain = max(float(gain), 0.0)

    caps = []
    if use_mi_cap:
        caps.append(entropy(p_n, eps=eps))
        caps.append(entropy(q, eps=eps))

    if use_g_entropy_cap:
        caps.append(lambda_g * entropy(g, eps=eps))

    if len(caps) > 0:
        gain = min(gain, min(caps))

    return float(gain)
```

Recommended ablations:

1. No cap.
2. Mutual-information cap only: \(\min\{H(\boldsymbol p),H(\boldsymbol q)\}\).
3. Response-collapse cap only: \(H(\boldsymbol g)\).
4. Combined cap: \(\min\{H(\boldsymbol p),H(\boldsymbol q),H(\boldsymbol g)\}\).

If the \(H(\boldsymbol g)\) cap helps most, the dominant failure mode is likely local response collapse rather than a violation of the mutual-information bound.

---

# Step 9: Empirical mean or UCB acquisition score

Repeat the sampling procedure \(S\) times to obtain gains

\[
G^{(1)}_{nm},\dots,G^{(S)}_{nm}.
\]

The empirical mean is

\[
\bar G_{nm}
=
\frac{1}{S}\sum_{s=1}^S G^{(s)}_{nm}.
\]

The empirical standard deviation is

\[
\hat\sigma_{nm}
=
\sqrt{
\frac{1}{S-1}
\sum_{s=1}^S
\left(G^{(s)}_{nm}-\bar G_{nm}\right)^2
}.
\]

The UCB score is

\[
\operatorname{UCB}_{nm}
=
\bar G_{nm}
+
\lambda_{\mathrm{ucb}}\hat\sigma_{nm}.
\]

Implementation:

```python

def empirical_ucb(gains, lambda_ucb=1.0):
    gains = np.asarray(gains, dtype=float)
    mean_gain = float(gains.mean())
    std_gain = float(gains.std(ddof=1)) if len(gains) > 1 else 0.0
    return mean_gain + lambda_ucb * std_gain, mean_gain, std_gain
```

Full sampling skeleton:

```python

def candidate_pair_score_likelihood_theta(
    candidate_idx,
    annotator_idx,
    observed_indices_m,
    z_im,
    w_ni,
    p_mean,
    p_candidate_mean,
    C,
    beta0,
    a_theta,
    b_theta,
    theta_grid,
    n_samples=10,
    lambda_ucb=1.0,
    use_mi_cap=True,
    use_g_entropy_cap=True,
    lambda_g=1.0,
    rng=None,
    eps=1e-12,
):
    if rng is None:
        rng = np.random.default_rng()

    beta_nm = local_response_bias_posterior(
        observed_indices_m=observed_indices_m,
        z_im=z_im,
        w_ni=w_ni,
        C=C,
        beta0=beta0,
    )

    gains = []
    theta_samples = []
    g_entropies = []

    for _ in range(n_samples):
        theta_s, g_s, theta_probs = sample_local_channel_parameters(
            beta_nm=beta_nm,
            theta_grid=theta_grid,
            observed_indices_m=observed_indices_m,
            z_im=z_im,
            w_ni=w_ni,
            p_mean=p_mean,
            a_theta=a_theta,
            b_theta=b_theta,
            rng=rng,
            eps=eps,
        )

        # Lightweight version: use deterministic candidate p.
        # If desired, replace this by a Dirichlet sample for p_n.
        p_n_s = np.asarray(p_candidate_mean, dtype=float)
        p_n_s = np.clip(p_n_s, eps, 1.0)
        p_n_s = p_n_s / p_n_s.sum()

        gain_s, q_s = sampled_information_gain(
            p_n=p_n_s,
            theta=theta_s,
            g=g_s,
            eps=eps,
        )

        gain_s = robust_cap_gain(
            gain=gain_s,
            p_n=p_n_s,
            q=q_s,
            g=g_s,
            use_mi_cap=use_mi_cap,
            use_g_entropy_cap=use_g_entropy_cap,
            lambda_g=lambda_g,
            eps=eps,
        )

        gains.append(gain_s)
        theta_samples.append(theta_s)
        g_entropies.append(entropy(g_s, eps=eps))

    score, mean_gain, std_gain = empirical_ucb(gains, lambda_ucb=lambda_ucb)

    diagnostics = {
        "mean_gain": mean_gain,
        "std_gain": std_gain,
        "score": score,
        "mean_theta": float(np.mean(theta_samples)),
        "std_theta": float(np.std(theta_samples, ddof=1)) if len(theta_samples) > 1 else 0.0,
        "mean_H_g": float(np.mean(g_entropies)),
        "beta_nm": beta_nm,
    }

    return score, diagnostics
```

---

# Diagnostics to log

For debugging spammers, log the following per candidate pair or averaged per annotator:

1. Local response-bias entropy:

\[
H(\boldsymbol g_{nm}).
\]

2. Predictive queried-label entropy:

\[
H(\boldsymbol q_{nm}).
\]

3. Candidate prior entropy:

\[
H(\boldsymbol p_n).
\]

4. Posterior mean or sampled mean of \(\theta_{nm}\).

5. Local comparison values:

\[
p_i^{(z_{im})}-g_{nm}^{(z_{im})}.
\]

If this value is often positive for spammers, the likelihood estimator may still overestimate \(\theta_{nm}\), usually because the classifier has learned the spammer bias or because \(\boldsymbol g_{nm}\) is not yet concentrated enough.

6. How often the response-collapse cap \(H(\boldsymbol g_{nm})\) is active.

If this cap is often active and improves performance, the main failure mode is local response collapse.

---

# Recommended paper wording

The local response-bias channel is

\[
K_{nm}(l\mid c)
=
\theta_{nm}\mathbb{1}[l=c]
+(1-\theta_{nm})g_{nm}^{(l)}.
\]

For each candidate pair \((n,m)\), the local response-bias distribution \(\boldsymbol g_{nm}\) is estimated from kernel-weighted label counts of annotator \(m\) in the neighborhood of \(x_n\). Conditional on \(\boldsymbol g_{nm}\), the task-dependence parameter \(\theta_{nm}\) is inferred from the likelihood of previously observed labels,

\[
\Pr(z_{im}\mid\boldsymbol p_i,\theta_{nm},\boldsymbol g_{nm})
=
\theta_{nm}p_i^{(z_{im})}
+(1-\theta_{nm})g_{nm}^{(z_{im})}.
\]

This likelihood compares whether an observed label is better explained by the classifier belief or by the annotator's local marginal response bias. We approximate the one-dimensional posterior over \(\theta_{nm}\) by grid evaluation and sample \(\theta_{nm}\) jointly with \(\boldsymbol g_{nm}\) to obtain a distribution over information gains.

For additional robustness, we optionally cap the estimated gain by the entropy of the local response-bias distribution \(H(\boldsymbol g_{nm})\). This term is not a mutual-information bound; it acts as a response-collapse regularizer that suppresses annotators whose local responses are nearly deterministic.

---

# Practical recommendation

Try this variant:

\[
\text{likelihood-based local }\theta_{nm}
+
\text{sampled local }\boldsymbol g_{nm}
+
\text{response-collapse cap }H(\boldsymbol g_{nm}).
\]

Ablate against:

1. old channel + information-gain cap,
2. response-bias channel + proxy \(\theta\),
3. response-bias channel + likelihood \(\theta\),
4. response-bias channel + likelihood \(\theta\) + \(H(\boldsymbol g)\) cap.

If variant 4 wins, the interpretation is clear: likelihood-based \(\theta\) improves local task-dependence estimation, while the \(H(\boldsymbol g)\) cap protects against residual response-collapse failures.
