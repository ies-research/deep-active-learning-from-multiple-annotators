# Local Response-Bias Mixture Channel for Instance-Dependent Annotator Modeling

## Goal

For each candidate instance--annotator pair \((n,m)\), estimate a local annotator-response channel

\[
K_{nm}(l \mid c)
=
\theta_{nm}\mathbb{1}[l=c]
+
(1-\theta_{nm})g_{nm}^{(l)} .
\]

Here:

- \(c \in \{1,\dots,C\}\) is the latent true class.
- \(l \in \{1,\dots,C\}\) is the label emitted by annotator \(m\).
- \(\boldsymbol{g}_{nm}\in\Delta_C\) is the local marginal response distribution of annotator \(m\) around candidate instance \(x_n\).
- \(\theta_{nm}\in[0,1]\) is the local task-dependence strength of annotator \(m\) around \(x_n\).

Interpretation:

- With probability \(\theta_{nm}\), the annotator emits the true class.
- With probability \(1-\theta_{nm}\), the annotator emits a label according to their local response bias \(\boldsymbol{g}_{nm}\).

This channel separates task-dependent information from local response bias. A deterministic local spammer with \(\theta_{nm}=0\) and \(g_{nm}^{(k)}=1\) becomes class-independent and therefore has zero mutual information with the true class.

---

## Inputs

Assume:

- `C`: number of classes.
- `candidate_idx = n`.
- `annotator_idx = m`.
- `observed_indices_m`: indices \(i\) previously labeled by annotator \(m\).
- `z_im[i]`: observed label from annotator \(m\) for instance \(i\), integer in `{0, ..., C-1}`.
- `w_ni[i]`: kernel similarity weight between candidate instance \(x_n\) and observed instance \(x_i\).
- `alpha_p[i, c]`: Dirichlet parameters for the classifier belief \(\boldsymbol{p}_i\), or alternatively deterministic classifier probabilities.
- `beta0`: Dirichlet prior pseudo-count for local response distribution.
- `a_theta, b_theta`: Beta prior parameters for \(\theta_{nm}\).
- `eps`: small numerical constant, e.g. `1e-12`.
- `min_denom`: threshold for identifying weakly informative observations, e.g. `1e-6`.
- Optional: `N_eff_max`: cap for effective sample size.

---

## Step 1: Estimate local response-bias distribution \(\boldsymbol{g}_{nm}\)

For candidate pair \((n,m)\), compute kernel-weighted Dirichlet counts:

\[
\beta_{nm}^{(l)}
=
\beta_0
+
\sum_{i\in\mathcal{D}_m}
w_{ni}\mathbb{1}[z_{im}=l].
\]

In code:

```python
beta_nm = beta0 * np.ones(C)

for i in observed_indices_m:
    label = z_im[i]
    beta_nm[label] += w_ni[i]
```

The local response-bias posterior is

\[
\boldsymbol{g}_{nm}
\sim
\operatorname{Dir}(\boldsymbol{\beta}_{nm}).
\]

Its posterior mean is

\[
\bar{g}_{nm}^{(l)}
=
\frac{\beta_{nm}^{(l)}}{\sum_{u=1}^C \beta_{nm}^{(u)}}.
\]

Use this posterior mean for the lightweight construction of the \(\theta_{nm}\) posterior.

```python
g_mean = beta_nm / beta_nm.sum()
```

---

## Step 2: Compute classifier belief moments

For each previously observed instance \(i\), we need:

\[
h_i = \mathbb{E}[p_i^{(z_{im})}],
\]

\[
r_i = \mathbb{E}[\|\boldsymbol{p}_i\|_2^2],
\]

and

\[
b_{ni}
=
\mathbb{E}[\boldsymbol{p}_i]^\top \mathbb{E}[\boldsymbol{g}_{nm}].
\]

If the classifier belief is deterministic, i.e. `p_mean[i]` is already a probability vector, use:

\[
h_i = p_i^{(z_{im})},
\]

\[
r_i = \sum_{c=1}^C (p_i^{(c)})^2,
\]

\[
b_{ni} = \boldsymbol{p}_i^\top \bar{\boldsymbol{g}}_{nm}.
\]

If the classifier belief is Dirichlet,

\[
\boldsymbol{p}_i \sim \operatorname{Dir}(\boldsymbol{\alpha}_i),
\]

then

\[
\mathbb{E}[p_i^{(c)}]
=
\frac{\alpha_i^{(c)}}{\alpha_{i,0}},
\qquad
\alpha_{i,0}=\sum_c\alpha_i^{(c)}.
\]

The second moment is

\[
\mathbb{E}\left[(p_i^{(c)})^2\right]
=
\frac{
\alpha_i^{(c)}(\alpha_i^{(c)}+1)
}{
\alpha_{i,0}(\alpha_{i,0}+1)
}.
\]

Therefore,

\[
r_i
=
\mathbb{E}[\|\boldsymbol{p}_i\|_2^2]
=
\sum_{c=1}^C
\frac{
\alpha_i^{(c)}(\alpha_i^{(c)}+1)
}{
\alpha_{i,0}(\alpha_{i,0}+1)
}.
\]

Assuming posterior independence between \(\boldsymbol{p}_i\) and \(\boldsymbol{g}_{nm}\),

\[
b_{ni}
=
\mathbb{E}[\boldsymbol{p}_i]^\top
\mathbb{E}[\boldsymbol{g}_{nm}].
\]

---

## Step 3: Compute baseline-corrected pseudo-\(\theta\) observations

The old soft correctness signal was

\[
h_i = p_i^{(z_{im})}.
\]

This is not enough, because it confounds task-dependent correctness with response bias.

Under the local response-bias mixture channel, the predictive response distribution is

\[
q_i^{(l)}
=
\theta_{nm}p_i^{(l)}
+
(1-\theta_{nm})g_{nm}^{(l)}.
\]

The expected classifier probability assigned to the emitted label is

\[
\mathbb{E}_{Z\sim q_i}[p_i^{(Z)}]
=
\theta_{nm}\|\boldsymbol{p}_i\|_2^2
+
(1-\theta_{nm})\boldsymbol{p}_i^\top\boldsymbol{g}_{nm}.
\]

Using the moment notation:

\[
\mathbb{E}_{Z\sim q_i}[p_i^{(Z)}]
=
\theta_{nm}r_i
+
(1-\theta_{nm})b_{ni}
=
b_{ni}
+
\theta_{nm}(r_i-b_{ni}).
\]

Solving for \(\theta_{nm}\) gives the pseudo-observation:

\[
\hat{\theta}_{im\mid n}
=
\frac{h_i-b_{ni}}{r_i-b_{ni}}.
\]

Use clipping:

\[
\hat{\theta}_{im\mid n}
\leftarrow
\operatorname{clip}
\left(
\hat{\theta}_{im\mid n},
0,
1
\right).
\]

Important: if

\[
|r_i-b_{ni}|
\]

is very small, the observation is weakly informative because response-bias behavior and task-dependent behavior produce almost the same moment. Do not force it through the estimator with a fake denominator. Either skip it or give it near-zero evidence weight.

Implementation:

```python
theta_pseudo = []
raw_weights = []

for i in observed_indices_m:
    label = z_im[i]

    # Get p_mean_i and r_i.
    # If using deterministic probabilities:
    p_mean_i = p_probs[i]  # shape [C]
    h_i = p_mean_i[label]
    r_i = np.sum(p_mean_i ** 2)

    # If using Dirichlet alpha_p:
    # alpha_i = alpha_p[i]
    # alpha0 = alpha_i.sum()
    # p_mean_i = alpha_i / alpha0
    # h_i = p_mean_i[label]
    # r_i = np.sum(alpha_i * (alpha_i + 1.0) / (alpha0 * (alpha0 + 1.0)))

    b_ni = np.dot(p_mean_i, g_mean)
    denom = r_i - b_ni

    if abs(denom) < min_denom:
        continue

    theta_i = (h_i - b_ni) / denom
    theta_i = np.clip(theta_i, 0.0, 1.0)

    evidence_weight = w_ni[i] * abs(denom)

    theta_pseudo.append(theta_i)
    raw_weights.append(evidence_weight)
```

---

## Step 4: Combine locality, informativeness, and ESS

The raw evidence weight is

\[
\tilde{w}_{ni}
=
w_{ni}|r_i-b_{ni}|.
\]

This combines:

- locality through \(w_{ni}\),
- informativeness through \(|r_i-b_{ni}|\).

Use effective sample size to scale the pseudo-counts:

\[
N_{nm}^{\mathrm{eff}}
=
\frac{
\left(\sum_i \tilde{w}_{ni}\right)^2
}{
\sum_i \tilde{w}_{ni}^2
}.
\]

Normalize the weights:

\[
\bar{w}_{ni}
=
\frac{\tilde{w}_{ni}}{\sum_j\tilde{w}_{nj}}.
\]

Then compute fractional Beta pseudo-counts:

\[
A_{nm}
=
N_{nm}^{\mathrm{eff}}
\sum_i
\bar{w}_{ni}
\hat{\theta}_{im\mid n},
\]

\[
B_{nm}
=
N_{nm}^{\mathrm{eff}}
\sum_i
\bar{w}_{ni}
(1-\hat{\theta}_{im\mid n}).
\]

If `N_eff_max` is provided, cap the ESS:

\[
N_{nm}^{\mathrm{eff}}
\leftarrow
\min(N_{nm}^{\mathrm{eff}}, N_{\max}).
\]

Implementation:

```python
theta_pseudo = np.asarray(theta_pseudo, dtype=float)
raw_weights = np.asarray(raw_weights, dtype=float)

if len(theta_pseudo) == 0 or raw_weights.sum() <= eps:
    A_nm = 0.0
    B_nm = 0.0
else:
    weight_sum = raw_weights.sum()
    norm_weights = raw_weights / weight_sum

    N_eff = (weight_sum ** 2) / (np.sum(raw_weights ** 2) + eps)

    if N_eff_max is not None:
        N_eff = min(N_eff, N_eff_max)

    theta_bar = np.sum(norm_weights * theta_pseudo)

    A_nm = N_eff * theta_bar
    B_nm = N_eff * (1.0 - theta_bar)
```

---

## Step 5: Approximate posterior over \(\theta_{nm}\)

Use a Beta approximation:

\[
\Theta_{nm}
\approx
\operatorname{Beta}
\left(
a_\theta + A_{nm},
b_\theta + B_{nm}
\right).
\]

Implementation:

```python
a_post = a_theta + A_nm
b_post = b_theta + B_nm
```

This is not an exact conjugate posterior. It is a moment-based Beta approximation using fractional pseudo-evidence.

---

## Step 6: Sample local channel parameters for gain computation

For each Monte Carlo sample \(s=1,\dots,S\):

Sample local response bias:

\[
\boldsymbol{g}_{nm}^{(s)}
\sim
\operatorname{Dir}(\boldsymbol{\beta}_{nm}).
\]

Sample local task-dependence strength:

\[
\theta_{nm}^{(s)}
\sim
\operatorname{Beta}(a_\theta+A_{nm}, b_\theta+B_{nm}).
\]

Optionally sample candidate class belief:

\[
\boldsymbol{p}_{n}^{(s)}
\sim
p(\boldsymbol{p}_n\mid x_n,\mathbf{I}).
\]

If using deterministic classifier probabilities, set:

\[
\boldsymbol{p}_{n}^{(s)} = \bar{\boldsymbol{p}}_n.
\]

The sampled response channel is:

\[
K_{nm}^{(s)}(l\mid c)
=
\theta_{nm}^{(s)}\mathbb{1}[l=c]
+
(1-\theta_{nm}^{(s)})g_{nm}^{(s,l)}.
\]

The predictive queried-label distribution is:

\[
q_{nm}^{(s,l)}
=
\sum_{c=1}^C
p_n^{(s,c)}K_{nm}^{(s)}(l\mid c).
\]

For this channel, this simplifies to:

\[
q_{nm}^{(s,l)}
=
\theta_{nm}^{(s)}p_n^{(s,l)}
+
(1-\theta_{nm}^{(s)})g_{nm}^{(s,l)}.
\]

After hypothetically observing label \(l\), update the class belief by Bayes' rule:

\[
\tilde{p}_{nm,l}^{(s,c)}
=
\frac{
p_n^{(s,c)}
\left[
\theta_{nm}^{(s)}\mathbb{1}[l=c]
+
(1-\theta_{nm}^{(s)})g_{nm}^{(s,l)}
\right]
}{
q_{nm}^{(s,l)}
}.
\]

Implementation:

```python
# For each sample:
g_s = np.random.dirichlet(beta_nm)
theta_s = np.random.beta(a_post, b_post)

# p_n_s: shape [C]
# either deterministic p_mean_n or sampled from Dirichlet

q_s = theta_s * p_n_s + (1.0 - theta_s) * g_s
q_s = np.clip(q_s, eps, 1.0)
q_s = q_s / q_s.sum()
```

Posterior update for each emitted label `l`:

```python
posteriors = np.zeros((C, C))  # rows: emitted label l, columns: true class c

for l in range(C):
    likelihood = (1.0 - theta_s) * g_s[l] * np.ones(C)
    likelihood[l] += theta_s

    unnorm = p_n_s * likelihood
    denom = unnorm.sum()

    if denom <= eps:
        posteriors[l] = p_n_s
    else:
        posteriors[l] = unnorm / denom
```

---

## Step 7: Compute sampled information gain

If the gain is KL-based,

\[
G(\boldsymbol{p},\tilde{\boldsymbol{p}})
=
\operatorname{KL}
\left(
\tilde{\boldsymbol{p}}
\| 
\boldsymbol{p}
\right),
\]

then the expected gain is mutual information:

\[
G_{nm}^{(s)}
=
\sum_{l=1}^C
q_{nm}^{(s,l)}
\operatorname{KL}
\left(
\tilde{\boldsymbol{p}}_{nm,l}^{(s)}
\|
\boldsymbol{p}_{n}^{(s)}
\right).
\]

Implementation:

```python
def kl_div(posterior, prior, eps=1e-12):
    posterior = np.clip(posterior, eps, 1.0)
    prior = np.clip(prior, eps, 1.0)
    return np.sum(posterior * (np.log(posterior) - np.log(prior)))

gain_s = 0.0
for l in range(C):
    gain_s += q_s[l] * kl_div(posteriors[l], p_n_s, eps=eps)
```

Equivalent faster formula:

\[
I(Y;Z)
=
H(\boldsymbol{q})
-
\sum_{c=1}^C p^{(c)}H(K(\cdot\mid c)).
\]

The posterior-update version is usually easier to verify first.

---

## Step 8: Optional information-theoretic cap

For exact mutual information,

\[
0
\leq
I(Y;Z)
\leq
\min\{H(\boldsymbol{p}_n),H(\boldsymbol{q}_{nm})\}.
\]

If the gain is approximated, project it onto the feasible interval:

\[
G_{nm}^{(s)}
\leftarrow
\min
\left\{
\max\{0,G_{nm}^{(s)}\},
\min\left[
H(\boldsymbol{p}_n^{(s)}),
H(\boldsymbol{q}_{nm}^{(s)})
\right]
\right\}.
\]

Implementation:

```python
def entropy(prob, eps=1e-12):
    prob = np.clip(prob, eps, 1.0)
    prob = prob / prob.sum()
    return -np.sum(prob * np.log(prob))

upper = min(entropy(p_n_s, eps), entropy(q_s, eps))
gain_s = min(max(gain_s, 0.0), upper)
```

---

## Step 9: Empirical UCB over sampled gains

Repeat the sampling procedure \(S\) times and obtain:

\[
G_{nm}^{(1)},\dots,G_{nm}^{(S)}.
\]

The empirical mean is:

\[
\bar{G}_{nm}
=
\frac{1}{S}\sum_{s=1}^S G_{nm}^{(s)}.
\]

The empirical standard deviation is:

\[
\hat{\sigma}_{nm}
=
\sqrt{
\frac{1}{S-1}
\sum_{s=1}^S
\left(
G_{nm}^{(s)}-\bar{G}_{nm}
\right)^2
}.
\]

The UCB score is:

\[
\operatorname{UCB}_{nm}
=
\bar{G}_{nm}
+
\lambda_{\mathrm{ucb}}\hat{\sigma}_{nm}.
\]

Implementation:

```python
gains = np.asarray(gains, dtype=float)

mean_gain = gains.mean()

if len(gains) > 1:
    std_gain = gains.std(ddof=1)
else:
    std_gain = 0.0

score = mean_gain + lambda_ucb * std_gain
```

---

## Summary of the lightweight local method

For each candidate pair \((n,m)\):

1. Estimate local response-bias posterior

\[
\boldsymbol{g}_{nm}
\sim
\operatorname{Dir}(\boldsymbol{\beta}_{nm})
\]

from kernel-weighted label counts of annotator \(m\).

2. Build a local Beta approximation for \(\theta_{nm}\) using baseline-corrected pseudo-observations:

\[
\hat{\theta}_{im\mid n}
=
\operatorname{clip}
\left(
\frac{
h_i-b_{ni}
}{
r_i-b_{ni}
},
0,
1
\right),
\]

where

\[
h_i=\mathbb{E}[p_i^{(z_{im})}],
\]

\[
b_{ni}=
\mathbb{E}[\boldsymbol{p}_i]^\top
\mathbb{E}[\boldsymbol{g}_{nm}],
\]

\[
r_i=
\mathbb{E}[\|\boldsymbol{p}_i\|_2^2].
\]

3. Weight pseudo-observations by locality and informativeness:

\[
\tilde{w}_{ni}
=
w_{ni}|r_i-b_{ni}|.
\]

4. Use ESS-scaled pseudo-counts:

\[
N_{nm}^{\mathrm{eff}}
=
\frac{
(\sum_i\tilde{w}_{ni})^2
}{
\sum_i\tilde{w}_{ni}^2
}.
\]

\[
A_{nm}
=
N_{nm}^{\mathrm{eff}}
\sum_i
\frac{\tilde{w}_{ni}}{\sum_j\tilde{w}_{nj}}
\hat{\theta}_{im\mid n}.
\]

\[
B_{nm}
=
N_{nm}^{\mathrm{eff}}
\sum_i
\frac{\tilde{w}_{ni}}{\sum_j\tilde{w}_{nj}}
(1-\hat{\theta}_{im\mid n}).
\]

5. Approximate:

\[
\Theta_{nm}
\approx
\operatorname{Beta}
(a_\theta+A_{nm}, b_\theta+B_{nm}).
\]

6. Sample \(\boldsymbol{g}_{nm}\), \(\theta_{nm}\), and optionally \(\boldsymbol{p}_n\), then compute information gain under:

\[
K_{nm}(l\mid c)
=
\theta_{nm}\mathbb{1}[l=c]
+
(1-\theta_{nm})g_{nm}^{(l)}.
\]

7. Use empirical mean or empirical UCB of the sampled gains.

---

## Important implementation notes

- Do not estimate one independent \(\theta\) per observed pair. Observed pairs provide pseudo-evidence for the candidate-local parameter \(\theta_{nm}\).
- If \(|r_i-b_{ni}|\) is close to zero, skip the observation or give it near-zero weight.
- Use posterior means of both \(\boldsymbol{p}_i\) and \(\boldsymbol{g}_{nm}\) when constructing the lightweight Beta approximation.
- If \(\boldsymbol{p}_i\) is Dirichlet-distributed, use \(\mathbb{E}[\|\boldsymbol{p}_i\|_2^2]\), not \(\|\mathbb{E}[\boldsymbol{p}_i]\|_2^2\).
- The Beta posterior over \(\theta_{nm}\) is approximate, not exact conjugacy.
- The local response-bias distribution \(\boldsymbol{g}_{nm}\) is not global. It is estimated with kernel weights around candidate \(x_n\), so the method remains instance-dependent.
- The information-theoretic cap is optional, but useful for numerical safety.
