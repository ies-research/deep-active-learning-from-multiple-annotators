Algorithm: Budget-Aware Local Agreement Scorer
==============================================

Working name
------------
`BudgetAwareLocalAgreementScorer`


Goal
----
For each candidate sample-annotator pair `(n, m)`, compute a utility `U[n, m]`
for querying annotator `m` on sample `x_n`.

The scorer estimates **classifier-conditioned annotator agreement**, not true
annotator correctness directly. It asks:

    How likely is annotator m to agree with confident current classifier beliefs
    near candidate sample n?

This distinction matters. If the classifier is confidently wrong, agreement can
rank a correct annotator too low. Benchmarks should therefore report downstream
classification performance and, where possible, separate diagnostic agreement
and correctness measures.


Implementation call path
------------------------
The scorer can be called directly, but in the active-learning experiment loop
the surrounding pipeline does additional work before `BudgetAwareLocalAgreementScorer`
receives inputs.

In `scripts/experiment.py`, each cycle:

    1. computes the current pair budget and remaining pair budget;
    2. builds `available_mask` from still-missing pair labels and feasible
       annotators;
    3. computes per-annotator remaining capacity after total caps;
    4. asks the assigner for `constraint_pressure`;
    5. projects future annotator budgets as `projected_annotator_budget`;
    6. selects candidate samples with the sample query strategy;
    7. calls the pair scorer on the selected samples and feasible annotators;
    8. passes the resulting utility matrix to the assigner.

The active `configs/experiment.yaml` currently selects:

    scorer@scorer.actual: agreement_global

so this scorer is used only when the experiment is launched with an override
such as:

    scorer@scorer.actual=budget_aware_local_agreement

or:

    scorer@scorer.actual=blga_gated_randomized_ucb

Important config distinction:

- The class constructor defaults are conservative and mostly ablation-oriented:
  `thompson`, one Thompson draw, `locality_mode="local"`,
  `responsive_combination="prior"`, `local_evidence_mode="knn"`,
  `bias_model_correction="none"`, and `use_rho_correction=True`.
- `configs/scorer/budget_aware_local_agreement.yaml` selects a stronger BLGA
  variant: Thompson with 5 draws, local kernel evidence, gated global/local
  combination, response-bias model averaging, confidence-weighted
  soft chance-corrected agreement, uniform response-bias counts,
  `gate_constraint_coupling="linear"`, and `use_rho_correction=False`.
- `configs/scorer/blga_gated.yaml` is the shared local gated BLGA base config.
  It uses 10 Thompson draws and the same kernel, confidence, soft
  chance-corrected, uniform response-bias, linear gate-coupling, and rho-disabled
  defaults.
- `configs/scorer/blga_gated_randomized_ucb.yaml` is the selected Study 1 BLGA
  candidate. It switches the shared base to randomized std-UCB while keeping
  the same evidence, agreement, bias, gate, kernel, and rho choices.

This matters for reasoning. The implementation supports several conceptual
models; a discussion of assumptions should always name the active config path.


Pipeline at a glance
--------------------
The full scorer pipeline is:

    1. get classifier probabilities and embeddings;
    2. encode observed annotator labels into classifier class indices;
    3. convert classifier probabilities and observed labels into fractional
       success/failure evidence;
    4. compute budget-aware neighborhood scales `k_star`;
    5. build the weak chance-level Beta prior;
    6. compute a pooled population agreement estimate and annotator-global
       responsive posteriors;
    7. compute candidate-local evidence in embedding space;
    8. combine global and local responsive evidence, either by a local prior or
       by an explicit gate;
    9. optionally average with a global response-bias model;
   10. convert posteriors into utilities by posterior mean, Thompson sampling,
       UCB, or randomized UCB;
   11. mask already-labeled or unavailable sample-annotator pairs with `nan`.

The conceptual core is therefore not "estimate annotator accuracy" in one
step. It is:

    classifier pseudo-label belief
        -> agreement evidence
        -> global shrinkage
        -> local evidence
        -> exploration-aware pair utility


Parameter reference
-------------------
Core scoring:

- `score_mode : {"mean", "thompson", "ucb", "randomized_ucb"}`
  - `"mean"` returns posterior means.
  - `"thompson"` samples posterior utilities and averages `thompson_samples`
    draws.
  - `"ucb"` returns deterministic posterior optimism using Beta standard
    deviations.
  - `"randomized_ucb"` uses UCB-style posterior standard deviations with
    random nonnegative optimism multipliers.
- `thompson_samples : int`
  - Number of Monte Carlo samples when `score_mode="thompson"`.
  - `1` is ordinary Thompson sampling.
  - Larger values reduce sampling variance and move toward posterior means.
- `ucb_mode : {"std"}`
  - Uncertainty bonus used when `score_mode="ucb"` or
    `score_mode="randomized_ucb"`.
  - `"std"` adds a multiple of the Beta posterior standard deviation.
- `global_exploration_weight : float`
  - Number of global posterior standard deviations added in UCB mode.
- `local_exploration_weight : float`
  - Number of local posterior standard deviations added in UCB mode.
- `random_ucb_values : array-like`
  - Discrete optimism multipliers for `score_mode="randomized_ucb"`.
  - The default grid is `[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]`.
- `random_ucb_probs : array-like`
  - Sampling probabilities for `random_ucb_values`.
  - Defaults to a conservative distribution concentrated near 0.0 and 0.1 with
    rare large optimism.
- `constraint_damps_global_exploration : bool`
  - If true, global UCB optimism is multiplied by
    `1 - constraint_pressure`.
- `gate_constraint_coupling : {"none", "linear"}`
  - `"none"` keeps the gated local-trust threshold independent of assignment
    pressure.
  - `"linear"` preserves the legacy threshold
    `k_star * (1 - constraint_pressure)`.
- `random_state`
  - Random source for Beta, Bernoulli, and Dirichlet draws.

Global/local structure:

- `locality_mode : {"local", "global"}`
  - `"local"` computes candidate-local evidence.
  - `"global"` ignores local evidence and broadcasts each annotator's global
    responsive posterior to all candidate samples.
- `responsive_combination : {"prior", "gated"}`
  - `"prior"` injects the annotator-global estimate into the local Beta
    posterior as pseudo-counts.
  - `"gated"` keeps global and local Beta posteriors separate and mixes them by
    an explicit local-trust gate `lambda[n, m]`.
- `gated_thompson_mode : {"weighted_average", "mixture_sample"}`
  - Used only when `score_mode="thompson"` and
    `responsive_combination="gated"`.
  - `"weighted_average"` samples global and local components and returns their
    convex combination.
  - `"mixture_sample"` samples a Bernoulli global/local model indicator from
    `lambda[n, m]`, preserving more Thompson variance.
- `evidence_weighting : {"confidence", "entropy", "margin", "uniform"}`
  - Controls how classifier uncertainty reduces classifier-conditioned evidence
    mass.
  - `"confidence"` uses only the top probability and is the current default.
  - `"entropy"` penalizes distributed probability mass across all classes.
  - `"margin"` penalizes ambiguity between the top two classes.
  - `"uniform"` disables classifier-confidence evidence weighting and gives
    every observed label unit evidence mass.
- `agreement_mode : {"argmax", "soft_chance_corrected", "soft_raw_probability"}`
  - Controls how the annotator's observed label splits evidence mass into
    Beta success and failure.
  - `"argmax"` preserves the original hard top-class comparison.
  - `"soft_chance_corrected"` rewards classifier-supported labels above
    random guessing.
  - `"soft_raw_probability"` uses `P[i, Y[i,m]]` directly.

Local evidence:

- `local_evidence_mode : {"knn", "kernel"}`
  - `"knn"` uses uniform nearest-neighbor evidence.
  - `"kernel"` uses RBF-weighted evidence with a full-dataset kth-neighbor
    bandwidth.
- `local_kernel_bandwidth_mode : {"full_kth"}`
  - Current kernel bandwidth rule.
  - Locality is defined by full-dataset geometry, not by sparse annotator labels.
- `use_rho_correction : bool`
  - If true, the coverage ratio `rho[n, m]` corrects the local posterior.
  - If false, `rho[n, m]` is still computed for diagnostics, but the posterior
    uses `rho_effective[n, m] = 1`.
- `normalize_embeddings : bool`
  - If true, L2-normalize classifier embeddings before distance computations.
- `metric : str`
  - Distance metric passed to `sklearn.metrics.pairwise_distances`.
- `exclude_self : bool`
  - If true, exclude candidate `n` from its own local/reference neighborhood
    when `n` is already in the pool.

Prior strengths:

- `base_prior_strength : float`
  - Strength of the weak chance-level base prior.
- `prior_mean_min : float`
  - Lower and upper clipping margin for agreement modes whose random-guess
    prior can become numerically degenerate.
- `pool_prior_scale : float`
  - Multiplier for population-to-annotator global shrinkage.
- `global_prior_k_multiplier : float`
  - Multiplier applied to `k_star` for global pool-to-annotator shrinkage.
- `local_bandwidth_k_multiplier : float`
  - Multiplier applied to `k_star` for kNN local evidence or kernel bandwidth
    ranks.
- `local_gate_k_multiplier : float`
  - Multiplier applied to `k_star` for the gated local-trust threshold and the
    prior-combination local prior strength.
- `local_prior_scale : float`
  - Multiplier for annotator-global-to-local shrinkage in
    `responsive_combination="prior"`.
- `local_prior_min : float`
  - Minimum annotator-global-to-local prior strength in
    `responsive_combination="prior"`.

Bias correction:

- `bias_model_correction : {"none", "model_average"}`
  - `"none"` uses only the responsive agreement model.
  - `"model_average"` mixes the responsive model with a global response-bias
    model for constant-label spammers or random guessers.
- `bias_response_weighting : {"evidence", "uniform"}`
  - Controls how observed labels update the response-bias label histogram.
  - `"evidence"` preserves the original confidence-weighted histogram.
  - `"uniform"` gives each observed response one count, independent of
    classifier confidence.

Other:

- `missing_label`
  - Missing-label marker. If `None`, use `clf.missing_label` when available and
    otherwise `np.nan`.
- `available_mask`
  - Pair-level feasibility mask passed at call time.
- `remaining_budget`
  - Remaining pair-budget horizon passed at call time. This controls
    `k_star`.
- `projected_annotator_budget`
  - Optional future-only per-annotator budget projection passed at call time.
  - If present, it overrides scalar `remaining_budget` for computing
    annotator-specific `k_star`.
- `constraint_pressure`
  - Batch-level scalar in `[0, 1]` passed at call time.
  - It dampens global UCB exploration when
    `constraint_damps_global_exploration=True`.
  - It affects the gated local-trust threshold only when
    `gate_constraint_coupling="linear"`.
- `store_neighbor_diagnostics : bool`
  - Stores neighbor-level arrays for visualization/debugging.
- `eps : float`
  - Numerical stability constant.


Notation
--------
Global dimensions:

    N = number of samples
    M = number of annotators
    C = number of classes
    D = embedding dimension

Inputs:

    X : samples, shape (N, ...)
    E : classifier embeddings, shape (N, D)
    P : classifier probabilities, shape (N, C)
    Y : annotator labels, shape (N, M)

Candidate sets:

    sample_indices = queried candidate sample ids
    annotator_indices = candidate annotator ids

For annotator `m`:

    O_m = {i : Y[i, m] is observed}
    L_m = |O_m|

The scorer returns a utility matrix in local candidate order:

    U[n, m]

where `n` indexes `sample_indices` and `m` indexes `annotator_indices`.
Infeasible pairs are returned as `np.nan`.


Step 1: Read classifier probabilities and embeddings
----------------------------------------------------
The scorer requires:

    P, E = clf.predict_proba(X, extra_outputs=["embeddings"])

with:

    P.shape == (N, C)
    E.shape[0] == N

The scorer also reads:

    clf.classes_
    clf.missing_label, if missing_label is not set on the scorer

Probabilities are clipped to nonnegative values and normalized row-wise:

    P[i, c] <- max(P[i, c], 0)
    P[i, :] <- P[i, :] / sum_c P[i, c]

Observed labels are encoded into class indices using `clf.classes_`. An
observed label not present in `clf.classes_` raises an error.


Step 2: Build classifier-conditioned agreement evidence
-------------------------------------------------------
For each sample `i`, compute an evidence weight:

    q[i] in [0, 1]

The selected `evidence_weighting` mode controls how classifier probabilities
become total Beta evidence mass.

Confidence weighting, the default:

    p_max[i] = max_c P[i, c]

    q[i] = (p_max[i] - 1 / C) / (1 - 1 / C)

Entropy weighting:

    H[i] = - sum_c P[i, c] * log(max(P[i, c], eps))

    q[i] = 1 - H[i] / log(C)

Margin weighting:

    p_1[i] = largest_c P[i, c]
    p_2[i] = second_largest_c P[i, c]

    q[i] = p_1[i] - p_2[i]

Uniform weighting:

    q[i] = 1

All modes clip the final weight:

    q[i] = clip(q[i], 0, 1)

Practical interpretation:

- `"confidence"` preserves the previous implementation. It only looks at the
  largest probability.
- `"entropy"` is stricter when probability mass is diffuse across many classes.
- `"margin"` is strict when the top two classes are close, even if the top class
  has moderate absolute probability.
- `"uniform"` deactivates confidence-based evidence scaling. This is useful for
  ablations where `agreement_mode` alone should determine success and failure.

For uniform probabilities, `"confidence"`, `"entropy"`, and `"margin"` give zero
or near-zero evidence. `"uniform"` gives `q = 1`.

Next compute label support:

Argmax agreement:

    y_hat[i] = argmax_c P[i, c]

    support[i, m] = 1[Y[i, m] == y_hat[i]]

Soft chance-corrected agreement:

    support_by_class[i, c] =
        clip((P[i, c] - 1 / C) / (1 - 1 / C), 0, 1)

    support[i, m] = support_by_class[i, Y[i, m]]

Soft raw-probability agreement:

    support[i, m] = P[i, Y[i, m]]

For every observed label:

    success[i, m] = q[i] * support[i, m]

    failure[i, m] = q[i] * (1 - support[i, m])

Unobserved labels contribute zero success and zero failure.

Interpretation:

- `agreement_mode="argmax"` estimates hard agreement with the classifier's
  selected top class.
- `agreement_mode="soft_raw_probability"` is closest to CROWDLAB-style
  self-confidence label quality.
- `agreement_mode="soft_chance_corrected"` estimates classifier support above
  random guessing. A label at chance probability gets zero success support; a
  label below chance is clipped to zero support and therefore contributes only
  failure under the available evidence mass.


Step 3: Compute budget-aware local scale `k_star`
-------------------------------------------------
Let:

    L_total = total number of observed labels in Y

If `remaining_budget` is absent:

    T = L_total / M
    k_star[m] = ceil(sqrt(T)) for all m

If `remaining_budget` is scalar `B`:

    T = (L_total + max(B, 0)) / M
    k_star[m] = ceil(sqrt(T)) for all m

If `remaining_budget` is per annotator, shape `(M,)`:

    T_m = L_m + max(B_m, 0)
    k_star[m] = ceil(sqrt(T_m))

If `projected_annotator_budget` is provided, shape `(M,)`, it takes precedence
over `remaining_budget` and is interpreted as future labels only:

    T_m = L_m + max(projected_annotator_budget[m], 0)
    k_star[m] = ceil(sqrt(T_m))

The experiment loop can compute this projection from current assigner
constraints using a utility-neutral water-fill over remaining feasible
annotator capacity. This makes locality scale constraint-aware without letting
the current BLGA utility matrix self-reinforce future scale estimates.

Interpretation:

- `k_star[m]` is the target local evidence scale for annotator `m`.
- Early in the run it is small, so local estimates remain cautious.
- Later it grows with available evidence.
- Scalar `remaining_budget` gives a shared scale across annotators.
- `projected_annotator_budget` gives annotator-specific scales when histories,
  caps, or remaining feasible labels differ.

The base scale is split into role-specific scales:

    k_global[m] = global_prior_k_multiplier * k_star[m]

    k_local[m] = ceil(local_bandwidth_k_multiplier * k_star[m])

    k_gate[m] = local_gate_k_multiplier * k_star[m]

Interpretation of the split:

- `k_global` controls population-to-annotator shrinkage.
- `k_local` controls kNN evidence counts or full-dataset kernel bandwidth ranks.
- `k_gate` controls the local-trust threshold in gated mode and local prior
  strength in prior-combination mode.


Step 4: Compute weak agreement prior
------------------------------------
For `agreement_mode="argmax"` and `agreement_mode="soft_raw_probability"`, use
the uniform-random label baseline:

    p0 = 1 / C

For `agreement_mode="soft_chance_corrected"`, use the current classifier's
global random-label expected support:

    support_by_class[i, c] =
        clip((P[i, c] - 1 / C) / (1 - 1 / C), 0, 1)

    p0 = mean_{i,c} support_by_class[i, c]

and clip it away from degenerate Beta parameters:

    p0 = clip(p0, prior_mean_min, 1 - prior_mean_min)

Then:

    alpha0 = base_prior_strength * p0
    beta0  = base_prior_strength * (1 - p0)

Example:

    C = 10
    base_prior_strength = 1

    alpha0 = 0.1
    beta0  = 0.9

For a uniform classifier distribution under `soft_chance_corrected`, the raw
random-label expected support is zero, so `prior_mean_min` supplies the weak
nondegenerate prior mean.


Step 5: Compute pool and annotator-global responsive posteriors
---------------------------------------------------------------
Pool evidence:

    S_pool = sum_{i,m} success[i, m]
    F_pool = sum_{i,m} failure[i, m]

Pool mean:

    mu_pool =
        (alpha0 + S_pool)
        / (alpha0 + beta0 + S_pool + F_pool)

Annotator evidence:

    S_m = sum_i success[i, m]
    F_m = sum_i failure[i, m]
    G_m = S_m + F_m

Population-to-annotator shrinkage strength:

    tau_pool[m] = pool_prior_scale * k_global[m]

Annotator-global mean:

    mu_global[m] =
        (tau_pool[m] * mu_pool + S_m)
        / (tau_pool[m] + G_m)

If the denominator is numerically zero:

    mu_global[m] = mu_pool

Annotator-global Beta posterior:

    alpha_G[m] = tau_pool[m] * mu_pool + S_m

    beta_G[m]  = tau_pool[m] * (1 - mu_pool) + F_m

This posterior is used directly in:

- `locality_mode="global"`;
- the global branch of `responsive_combination="gated"`;
- response-bias model comparison diagnostics.


Step 6: Compute local evidence
------------------------------
Local evidence is computed only when:

    locality_mode = "local"

If:

    locality_mode = "global"

then the scorer skips local evidence and uses:

    alpha[n, m] = alpha_G[m]
    beta[n, m]  = beta_G[m]

for every candidate sample `n`.


Step 6a: Uniform kNN local evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used when:

    local_evidence_mode = "knn"

For annotator `m`:

    k_m = min(k_local[m], L_m)

If `k_m <= 0`, then:

    S_local[n, m] = 0
    F_local[n, m] = 0
    h_actual[n, m] = nan
    h_ref[n, m] = nan
    rho[n, m] = 1

Otherwise compute full-dataset distances from each candidate `n`:

    d_full[n, j] = d(E[n], E[j])

If `exclude_self=True`, set `d_full[n, n] = inf`.

Expected full-dataset reference rank:

    R_ref[m] = ceil(k_m * N / L_m)
    R_ref[m] = clip(R_ref[m], 1, max(N - 1, 1))

Reference bandwidth:

    h_ref[n, m] = R_ref[m]-th smallest finite value of d_full[n, :]

Observed-label distances:

    d_obs[n, i] = d(E[n], E[i]) for i in O_m

If `exclude_self=True`, candidate self-label distance is set to `inf`.

Let `N_k(n, m)` be the `row_k` nearest finite observed labels, where:

    row_k = min(k_m, number of finite observed distances)

Actual observed bandwidth:

    h_actual[n, m] = distance to the row_k-th nearest observed label

Uniform local evidence:

    S_local[n, m] = sum_{i in N_k(n,m)} success[i, m]

    F_local[n, m] = sum_{i in N_k(n,m)} failure[i, m]

Coverage ratio:

    rho[n, m] = h_actual[n, m] / (h_ref[n, m] + eps)

Effective coverage ratio used by the posterior:

    rho_effective[n, m] = rho[n, m] if use_rho_correction else 1

Interpretation:

- `rho < 1`: observed labels are denser than random-spread coverage.
- `rho ~= 1`: observed labels are about as broad as random-spread coverage.
- `rho > 1`: observed labels are broader than expected.

Important limitation:

`rho ~= 1` does not prove useful locality. If annotator `m` has only a few
labels among thousands, the random-spread reference radius is already broad.
Then `rho` can be near 1 even though the evidence is practically global.


Step 6b: Kernel local evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Used when:

    local_evidence_mode = "kernel"
    local_kernel_bandwidth_mode = "full_kth"

The kernel mode defines locality from the full dataset geometry, not from sparse
annotator labels.

Full-dataset local bandwidth:

    k_full[m] = clip(k_local[m], 1, max(N - 1, 1))

    sigma_full[n, m] =
        k_full[m]-th nearest finite full-dataset distance from E[n]

The implementation stores:

    h_ref[n, m] = sigma_full[n, m]
    last_local_kernel_bandwidth_[n, m] = sigma_full[n, m]

Observed-label coverage diagnostic:

    row_k = min(k_local[m], number of finite observed-label distances)

    sigma_obs[n, m] =
        row_k-th nearest finite observed-label distance from E[n]

The implementation stores:

    h_actual[n, m] = sigma_obs[n, m]

Coverage ratio:

    rho[n, m] = sigma_obs[n, m] / (sigma_full[n, m] + eps)

Effective coverage ratio used by the posterior:

    rho_effective[n, m] = rho[n, m] if use_rho_correction else 1

RBF weights over all finite observed labels of annotator `m`:

    w_i[n, m] =
        exp(-0.5 * (d(E[n], E[i]) / max(sigma_full[n, m], eps))^2)

Weighted local evidence:

    S_local[n, m] = sum_{i in O_m} w_i[n, m] * success[i, m]

    F_local[n, m] = sum_{i in O_m} w_i[n, m] * failure[i, m]

Kernel weight mass:

    W_local[n, m] = sum_{i in O_m} w_i[n, m]

The implementation stores:

    last_local_kernel_weight_sum_[n, m] = W_local[n, m]

Interpretation:

- Far-away sparse annotator labels are not forced into the local neighborhood.
- If annotator `m` has only far-away labels, `W_local`, `S_local`, and
  `F_local` become small.
- The global fallback then comes from the prior/gate rather than from broad
  local evidence.


Step 7: Combine global and local responsive estimates
-----------------------------------------------------
Define local effective mass:

    M_local[n, m] = S_local[n, m] + F_local[n, m]


Step 7a: Prior combination
~~~~~~~~~~~~~~~~~~~~~~~~~~
Used when:

    responsive_combination = "prior"

Base global-to-local prior strength:

    nu_base[m] =
        max(
            local_prior_min,
            local_prior_scale * min(k_gate[m], G_m)
        )

Radius-aware prior strength:

    nu[n, m] = nu_base[m] * max(1, rho_effective[n, m])

Local Beta posterior:

    alpha[n, m] =
        nu[n, m] * mu_global[m] + S_local[n, m]

    beta[n, m] =
        nu[n, m] * (1 - mu_global[m]) + F_local[n, m]

Posterior mean:

    raw_score[n, m] =
        alpha[n, m] / (alpha[n, m] + beta[n, m])

If local evidence is zero:

    raw_score[n, m] = mu_global[m]

Interpretation:

- The annotator-global estimate is a local Beta prior.
- Local evidence competes additively with prior strength `nu`.
- `rho_effective > 1` increases global shrinkage when local evidence is broad.
- If `use_rho_correction=False`, broadness is only diagnostic and does not
  change `nu`.


Step 7b: Gated combination
~~~~~~~~~~~~~~~~~~~~~~~~~~
Used when:

    responsive_combination = "gated"

Global Beta posterior for each pair:

    alpha_G_pair[n, m] = alpha_G[m]
    beta_G_pair[n, m]  = beta_G[m]

Local Beta posterior with only the weak chance prior:

    alpha_L[n, m] = alpha0 + S_local[n, m]

    beta_L[n, m]  = beta0 + F_local[n, m]

Means:

    mean_G[n, m] =
        alpha_G_pair[n, m] / (alpha_G_pair[n, m] + beta_G_pair[n, m])

    mean_L[n, m] =
        alpha_L[n, m] / (alpha_L[n, m] + beta_L[n, m])

Local-trust gate:

If:

    gate_constraint_coupling = "none"

then:

    k_effective[n, m] = max(eps, k_gate[m])

If:

    gate_constraint_coupling = "linear"

then:

    k_effective[n, m] =
        max(eps, k_gate[m] * (1 - constraint_pressure))

For both gate modes:

    mass_gate[n, m] =
        M_local[n, m] / (M_local[n, m] + k_effective[n, m])

    radius_gate[n, m] =
        min(1, 1 / max(rho_effective[n, m], eps))

    lambda[n, m] =
        clip(mass_gate[n, m] * radius_gate[n, m], 0, 1)

Gated posterior mean:

    raw_score[n, m] =
        (1 - lambda[n, m]) * mean_G[n, m]
      + lambda[n, m]       * mean_L[n, m]

Interpretation:

- If `gate_constraint_coupling="none"`, the local-trust gate is purely
  evidence-based: `M_local = k_gate` gives 50 percent local trust when
  `rho = 1`.
- If `gate_constraint_coupling="linear"`, higher `constraint_pressure` reduces
  the evidence threshold for trusting locality. This preserves the previous
  pragmatic coupling but mixes assignment pressure into the epistemic gate.
- `M_local = 0` gives full global fallback.
- `rho_effective > 1` explicitly reduces local trust.
- If `use_rho_correction=False`, local trust depends on local evidence mass but
  not on the coverage-ratio penalty.
- The global and local posteriors remain separate, which makes diagnostics and
  Thompson sampling more interpretable.

Constraint pressure:

- `constraint_pressure = 0` means the batch assignment can still concentrate on
  one annotator.
- `constraint_pressure = 1` means the assignment constraints force use of every
  feasible annotator.
- For the exact and greedy cap-based assigners, pressure is computed from the
  minimum number of annotators needed to spend the batch budget under current
  per-annotator upper bounds:

      pressure =
          (min_annotators_needed - 1) / (n_feasible_annotators - 1)

- With `gate_constraint_coupling="none"`, constraint pressure affects
  exploration policy but not local-trust evidence. With the selected
  `gate_constraint_coupling="linear"` setting, loose caps only moderately
  increase local trust, while exact quota-like constraints increase it strongly.


Step 8: Optional response-bias model averaging
----------------------------------------------
Used when:

    bias_model_correction = "model_average"

Disabled when:

    bias_model_correction = "none"

The model averaging compares two global explanations for annotator `m`.

Responsive model:

    theta_m ~ Beta(alpha0, beta0)

For `agreement_mode="argmax"`, this corresponds to the hard top-class
likelihood:

    P(Y[i, m] = y_hat[i] | theta_m) = theta_m

    P(Y[i, m] = c != y_hat[i] | theta_m) =
        (1 - theta_m) / (C - 1)

For soft agreement modes, there is no single hard success event. The
implementation keeps the same Beta evidence summary and lets the selected
`agreement_mode` define fractional `success[i, m]` and `failure[i, m]`.

Response-bias model:

    r_m ~ Dirichlet(eta)

    eta[c] = 1 / C

    P(Y[i, m] = c | r_m) = r_m[c]

Responsive weighted sufficient statistics use the selected `agreement_mode`:

    S_m = sum_i success[i, m]

    F_m = sum_i failure[i, m]

Response-bias label histogram:

If:

    bias_response_weighting = "evidence"

then:

    N_m[c] = sum_i q[i] * 1[Y[i, m] == c]

If:

    bias_response_weighting = "uniform"

then:

    N_m[c] = sum_i 1[Y[i, m] == c]

    Q_m = sum_c N_m[c]

Responsive log marginal likelihood:

    log_L_resp[m] =
        log Beta(alpha0 + S_m, beta0 + F_m)
      - log Beta(alpha0, beta0)
      - F_m * log(C - 1)

The final `-F_m * log(C - 1)` term is exact for hard argmax failures. With soft
agreement modes, `F_m` is fractional; the same term is retained as the
comparable non-top-class penalty used by the implemented response-bias model
comparison.

Response-bias log marginal likelihood:

    eta_sum = sum_c eta[c]

    log_L_bias[m] =
        log Gamma(eta_sum)
      - log Gamma(eta_sum + Q_m)
      + sum_c [
            log Gamma(eta[c] + N_m[c])
          - log Gamma(eta[c])
        ]

Equal model priors are assumed. Posterior model probabilities are:

    log_norm[m] =
        logsumexp(log_L_resp[m], log_L_bias[m])

    p_resp[m] =
        exp(log_L_resp[m] - log_norm[m])

    p_bias[m] =
        1 - p_resp[m]

Response-bias posterior mean:

    E[r_m[c]] =
        (eta[c] + N_m[c]) / (eta_sum + Q_m)

Bias-branch score for candidate `n`:

    bias_score[n, m] =
        sum_c P[n, c] * E[r_m[c]]

Mean utility with model averaging:

    U_mean[n, m] =
        p_resp[m] * raw_score[n, m]
      + p_bias[m] * bias_score[n, m]

Interpretation:

- The responsive model explains annotators who track the classifier's
  sample-dependent supported labels.
- The response-bias model explains annotators whose labels are better described
  by a sample-independent class-response distribution.
- No hard spammer threshold is needed.


Step 9: Thompson utility sampling
---------------------------------
Used when:

    score_mode = "thompson"

Let:

    R = thompson_samples


Step 9a: Responsive sampling for prior/global mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For `responsive_combination="prior"` or `locality_mode="global"`:

    theta_r[n, m] ~ Beta(alpha[n, m], beta[n, m])


Step 9b: Responsive sampling for gated mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For `responsive_combination="gated"`:

    theta_G,r[n, m] ~ Beta(alpha_G_pair[n, m], beta_G_pair[n, m])

    theta_L,r[n, m] ~ Beta(alpha_L[n, m], beta_L[n, m])

If:

    gated_thompson_mode = "weighted_average"

then:

    theta_r[n, m] =
        (1 - lambda[n, m]) * theta_G,r[n, m]
      + lambda[n, m]       * theta_L,r[n, m]

If:

    gated_thompson_mode = "mixture_sample"

then:

    z_local,r[n, m] ~ Bernoulli(lambda[n, m])

    theta_r[n, m] =
        theta_L,r[n, m] if z_local,r[n, m] = 1
        theta_G,r[n, m] otherwise

Interpretation:

- `"weighted_average"` has lower variance and is more exploitative.
- `"mixture_sample"` preserves model-choice uncertainty and is more exploratory.
- Both have the same gated posterior mean in expectation.


Step 9c: Add response-bias model averaging to Thompson draws
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If:

    bias_model_correction = "none"

then final utility is:

    U[n, m] = mean_r theta_r[n, m]

If:

    bias_model_correction = "model_average"

then for each draw `r` and annotator `m`:

    z_resp,r[m] ~ Bernoulli(p_resp[m])

If `z_resp,r[m] = 1`:

    U_r[n, m] = theta_r[n, m]

If `z_resp,r[m] = 0`:

    r_r[m] ~ Dirichlet(eta + N_m)

    U_r[n, m] =
        sum_c P[n, c] * r_r[m, c]

Final utility:

    U[n, m] = mean_r U_r[n, m]

Important detail:

`z_resp,r[m]` is sampled once per annotator per draw, then broadcast across
candidate samples. The response-bias model is annotator-global, not pair-local.


Step 10: UCB and randomized-UCB utility
---------------------------------------
Used when:

    score_mode = "ucb"

or:

    score_mode = "randomized_ucb"

Current implementation:

    ucb_mode = "std"

For any Beta posterior:

    mean = alpha / (alpha + beta)

    std =
        sqrt(
            alpha * beta
            / ((alpha + beta)^2 * (alpha + beta + 1))
        )

The exploration weights and random multipliers are measured in posterior
standard deviations.


Step 10a: UCB for prior/global mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For `responsive_combination="prior"` or `locality_mode="global"`:

    if locality_mode = "global":
        weight = global_exploration_weight
    else:
        weight = local_exploration_weight

If:

    constraint_damps_global_exploration = True
    locality_mode = "global"

then:

    weight = weight * (1 - constraint_pressure)

Responsive UCB:

    responsive_ucb[n, m] =
        clip(mean[n, m] + weight * std[n, m], 0, 1)

For `score_mode="randomized_ucb"`:

- in global locality mode, draw one nonnegative multiplier `z_G[m]` per
  annotator column;
- in local prior mode, draw one nonnegative local multiplier `z_L[n, m]` per
  pair.

Then multiply the corresponding deterministic exploration weight by `z`.


Step 10b: UCB for gated mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For `responsive_combination="gated"`:

    std_G[n, m] = std(Beta(alpha_G_pair[n, m], beta_G_pair[n, m]))
    std_L[n, m] = std(Beta(alpha_L[n, m], beta_L[n, m]))

Global exploration weight:

    w_G = global_exploration_weight

If:

    constraint_damps_global_exploration = True

then:

    w_G = w_G * (1 - constraint_pressure)

Local exploration weight:

    w_L = local_exploration_weight

For deterministic UCB:

    z_G[m] = 1
    z_L[n, m] = 1

For randomized UCB:

    z_G[m] ~ Discrete(random_ucb_values, random_ucb_probs)

    z_L[n, m] ~ Discrete(random_ucb_values, random_ucb_probs)

where `z_G[m]` is sampled once per annotator column and `z_L[n, m]` is sampled
per pair. The global draw is coherent across candidate samples for the same
annotator; the local draw targets within-sample annotator diversity.

Optimistic components:

    ucb_G[n, m] =
        mean_G[n, m] + w_G * z_G[m] * std_G[n, m]

    ucb_L[n, m] =
        mean_L[n, m] + w_L * z_L[n, m] * std_L[n, m]

Gated responsive UCB:

    responsive_ucb[n, m] =
        clip(
            (1 - lambda[n, m]) * ucb_G[n, m]
          + lambda[n, m]       * ucb_L[n, m],
            0,
            1
        )

Interpretation:

- Constraints substitute for global annotator exploration, so global optimism
  is damped by `constraint_pressure`.
- Constraints do not substitute for local-specialization exploration, so local
  optimism is not damped.
- Deterministic UCB is reproducible but can lock onto the same annotators in
  weakly constrained batch settings.
- Randomized UCB keeps the posterior standard deviation closed form and
  randomizes the optimism policy, rather than adding Monte Carlo noise to the
  estimated standard deviation.


Step 10c: Add response-bias model averaging to UCB/randomized-UCB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If:

    bias_model_correction = "none"

then final utility is:

    U[n, m] = responsive_ucb[n, m]

If:

    bias_model_correction = "model_average"

then:

    U[n, m] =
        p_resp[m] * responsive_ucb[n, m]
      + p_bias[m] * bias_score[n, m]

No response-bias uncertainty bonus is added in the current UCB/randomized-UCB
implementation.


Step 11: Apply feasibility mask
-------------------------------
After utility computation:

    feasible[n, m] =
        label Y[sample_indices[n], annotator_indices[m]] is not observed

If `available_mask` is provided:

    feasible[n, m] =
        feasible[n, m] AND available_mask[n, m]

Final output:

    U[n, m] = computed utility if feasible[n, m]
    U[n, m] = np.nan otherwise


Assumption audit and reasoning
------------------------------
This section is the practical discussion checklist: what each step assumes, why
the assumption is reasonable, and where it can break.

Step 1, classifier probabilities and embeddings:

The scorer assumes the current classifier has enough signal for its
probabilities and embeddings to be useful. This is reasonable after the model
has seen enough labels because the active learner is trying to spend labels near
the classifier's current decision structure. It is weakest in very early cycles,
after distribution shift, or when the embedding space clusters by nuisance
features instead of the annotation skill structure.

Step 2, classifier-conditioned agreement evidence:

The evidence construction assumes that confident classifier beliefs should
count more than uncertain beliefs. That is reasonable if confidence correlates
with correctness or at least with stable local structure. The failure mode is
confident classifier error: then a good annotator who disagrees with the model
is treated as low agreement. `soft_chance_corrected` and `soft_raw_probability`
reduce some hard-argmax brittleness, but they do not remove dependence on the
classifier.

Step 3, budget-aware `k_star`:

The square-root rule assumes the useful local neighborhood should grow
sublinearly with the total number of labels expected for an annotator. This is a
standard bias-variance compromise for local smoothing: more labels permit a
larger neighborhood, but locality should not expand linearly. It is still a
heuristic. The most important checks are sensitivity to
`local_bandwidth_k_multiplier`, `local_gate_k_multiplier`, and whether
`projected_annotator_budget` changes conclusions under tight caps.

Step 4, weak chance-level prior:

The base Beta prior assumes an uninformative annotator agrees at roughly random
label chance before seeing evidence. This is reasonable as a weak stabilizer and
prevents zero-evidence pairs from becoming numerically degenerate. It can be
misleading under strong class imbalance or non-uniform response patterns, which
is one reason the optional response-bias model exists.

Step 5, pooled and annotator-global shrinkage:

The pooled estimate assumes annotators share enough behavior that population
agreement is a useful early prior for a specific annotator. This is reasonable
when annotator histories are sparse. It can wash out specialists, adversarial
annotators, or heterogeneous groups. The shrinkage strength `tau_pool` should
therefore be treated as a regularization parameter, not a fact about the data.

Step 6, local evidence:

The local step assumes annotator behavior is smooth in the classifier embedding
space: labels near a candidate are informative for that candidate. This is
reasonable for domain or feature specialists and for datasets where embeddings
capture semantic neighborhoods. It is weak for class-only expertise,
instruction-following effects, temporal drift, or embeddings dominated by
irrelevant visual/textual features.

Step 6a, kNN local evidence:

Uniform kNN assumes the nearest `k_local` observed labels are all equally local.
This is simple and works when annotator labels are dense enough around each
candidate. It is risky with sparse labels because the kth observed label may be
very far away; the method can then treat broad evidence as local unless the
`rho` correction is active.

Step 6b, kernel local evidence:

Kernel evidence assumes the full-dataset kth-neighbor radius is the right
locality scale, independent of how sparse an annotator's labels are. This is a
good default for BLGA because far-away sparse labels receive tiny RBF weights
instead of being forced into the neighborhood. The risk is bandwidth quality:
high-dimensional distances, duplicated embeddings, or uneven density can make
the full-dataset radius too small or too large.

Step 7a, prior combination:

The prior-combination branch assumes the annotator-global mean can be injected
as local pseudo-counts. This is coherent and easy to interpret: local evidence
adds to a global prior. It also couples global and local uncertainty into a
single Beta posterior, so diagnostics are less separated than in gated mode.
If `use_rho_correction=False`, broadness no longer increases shrinkage.

Step 7b, gated combination:

The gated branch assumes local trust should increase with local evidence mass
and decrease when the observed neighborhood radius is broader than the
reference radius. This is a reasonable heuristic model average between
"annotator has a global agreement level" and "annotator has local behavior
near this candidate." It is not a fully Bayesian posterior over local
specialization. `gate_constraint_coupling="none"` is cleaner epistemically and
is kept as an ablation; the selected `"linear"` setting intentionally lets
assignment pressure make the model trust local evidence sooner.

Step 8, response-bias model averaging:

The bias branch assumes some annotators are better explained by a
sample-independent label histogram than by sample-dependent agreement. This is
reasonable for constant-label spammers and random guessers. It can also absorb
real class preference or class-specialist behavior, so `p_bias` should be read
as "response-bias explanation is plausible", not as a moral spammer label.

Step 9, Thompson utilities:

Thompson sampling assumes posterior uncertainty should drive exploration
through random utility draws. This is reasonable for active learning because it
avoids always selecting the current posterior mean maximum. Averaging many
draws reduces variance and moves toward exploitation; `mixture_sample` keeps
more model-choice uncertainty than `weighted_average` in gated mode.

Step 10, UCB and randomized UCB:

UCB assumes the Beta posterior standard deviation is an adequate uncertainty
bonus. This is interpretable and deterministic, and randomized UCB adds
controlled policy randomness without changing the posterior model. The current
UCB implementation does not add a separate uncertainty bonus for the
response-bias branch, so bias-model uncertainty is less explored than
responsive-model uncertainty.

Step 11, feasibility masking:

The final mask assumes the assigner can treat `nan` as impossible and optimize
only over feasible pairs. This is the right separation of concerns: the scorer
estimates utility, while the assigner enforces pair feasibility and batch
constraints. The practical check is that each cycle still has enough non-`nan`
pairs to spend the requested budget or that the assigner's relaxation behavior
is intentional.

Overall reasonableness:

The approach is reasonable when the current classifier is a useful proxy,
embeddings capture locality relevant to annotator behavior, and the experiment
reports downstream task quality rather than only scorer agreement. The most
important assumptions to stress-test are classifier-conditioned agreement,
embedding-space locality, the square-root budget scale, the gated `lambda`
heuristic, and whether the response-bias branch is separating spammers from
real specialists.


Diagnostics
-----------
Pair-shaped diagnostics have shape:

    (len(sample_indices), len(annotator_indices))

Core pair diagnostics:

    last_alpha_
    last_beta_
    last_raw_score_
    last_final_score_
    last_h_actual_
    last_h_ref_
    last_rho_
    last_rho_effective_
    last_nu_
    last_k_local_
    last_local_success_
    last_local_failure_

Sample-shaped diagnostics have shape `(N,)`:

    last_evidence_weight_

Mode diagnostics:

    last_responsive_combination_
    last_gated_thompson_mode_
    last_ucb_mode_
    last_global_exploration_weight_
    last_local_exploration_weight_
    last_random_ucb_values_
    last_random_ucb_probs_
    last_random_ucb_global_multiplier_
    last_random_ucb_local_multiplier_
    last_constraint_damps_global_exploration_
    last_gate_constraint_coupling_
    last_evidence_weighting_
    last_agreement_mode_
    last_bias_response_weighting_
    last_constraint_pressure_
    last_local_evidence_mode_
    last_use_rho_correction_
    last_projected_annotator_budget_
    last_k_star_
    last_k_star_global_
    last_k_star_local_
    last_k_star_gate_

Global responsive diagnostics:

    last_mu_pool_
    last_mu_global_
    last_alpha_global_
    last_beta_global_
    last_tau_pool_

Gated diagnostics:

    last_lambda_local_
    last_gate_k_effective_
    last_alpha_local_
    last_beta_local_

Kernel diagnostics:

    last_local_kernel_bandwidth_
    last_local_kernel_weight_sum_

Bias model diagnostics:

    last_p_responsive_
    last_p_bias_
    last_log_likelihood_responsive_
    last_log_likelihood_bias_
    last_bias_score_
    last_bias_response_counts_

Neighbor diagnostics when `store_neighbor_diagnostics=True`:

    last_neighbor_indices_
    last_neighbor_distances_
    last_neighbor_success_
    last_neighbor_failure_
    last_neighbor_confidence_


Numerical examples
------------------

Example 1: Prior combination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Suppose:

    mu_global[m] = 0.70
    nu[n, m] = 2
    S_local[n, m] = 2.12
    F_local[n, m] = 0.54

Then:

    alpha = 2 * 0.70 + 2.12 = 3.52

    beta = 2 * 0.30 + 0.54 = 1.14

Posterior mean:

    raw_score = 3.52 / (3.52 + 1.14) = 0.755

Thompson sampling draws:

    theta ~ Beta(3.52, 1.14)


Example 2: Gated combination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Suppose:

    mean_G = 0.80
    mean_L = 0.55
    M_local = 4
    k_gate = 4
    rho = 1
    constraint_pressure = 0
    gate_constraint_coupling = "none"

Then:

    mass_gate = 4 / (4 + 4) = 0.5

    radius_gate = min(1, 1 / 1) = 1

    lambda = 0.5

Gated mean:

    raw_score = 0.5 * 0.80 + 0.5 * 0.55 = 0.675

If `rho = 2` instead:

    radius_gate = 1 / 2
    lambda = 0.25

    raw_score = 0.75 * 0.80 + 0.25 * 0.55 = 0.7375

Broad evidence therefore pulls the score back toward the global estimate.

If the same batch has `constraint_pressure = 0.5`, then:

    k_effective = 4

and the gated mean is unchanged because `gate_constraint_coupling="none"`.

If `gate_constraint_coupling="linear"` instead, then:

    k_effective = 4 * (1 - 0.5) = 2
    mass_gate = 4 / (4 + 2) = 0.6667

    lambda = 0.6667

    raw_score = 0.3333 * 0.80 + 0.6667 * 0.55 = 0.6333

The linear coupling makes higher constraint pressure move the gated mean toward
the local estimate earlier.


Example 3: Kernel mode does not force sparse far labels to be local
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One-dimensional embeddings:

    E = [0, 1, 2, 3, 4, 100, 101]

Candidate:

    n = 2, E[n] = 2

Annotator `m` has labels only at:

    E[5] = 100
    E[6] = 101

Let:

    k_star[m] = 2

Full-dataset bandwidth:

    distances from 2 to full data excluding self:
        1, 1, 2, 2, 98, 99

    sigma_full = 2nd nearest = 1

Observed-label diagnostic:

    observed distances:
        98, 99

    sigma_obs = 2nd nearest observed = 99

Coverage ratio:

    rho = 99 / (1 + eps) ~= 99

Kernel weights:

    w_5 = exp(-0.5 * 98^2) ~= 0
    w_6 = exp(-0.5 * 99^2) ~= 0

So:

    S_local ~= 0
    F_local ~= 0

The local model falls back to global, instead of treating far-away labels as
local evidence.


Example 4: Response-bias model correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Three classes:

    C = 3
    alpha0 = 1 / 3
    beta0 = 2 / 3
    eta = [1/3, 1/3, 1/3]

Observed exposure:

    agreement_mode:          argmax
    evidence_weighting:      uniform
    bias_response_weighting: uniform
    argmax labels:           0, 1, 2, 0, 1, 2
    annotator labels:        0, 0, 0, 0, 0, 0
    q:                       1, 1, 1, 1, 1, 1

Statistics:

    S_m = 2
    F_m = 4
    N_m = [6, 0, 0]
    Q_m = 6

Responsive log marginal:

    log_L_resp =
        log Beta(1/3 + 2, 2/3 + 4)
      - log Beta(1/3, 2/3)
      - 4 * log(2)

    log_L_resp = -7.7773

Bias log marginal:

    log_L_bias =
        log Gamma(1)
      - log Gamma(7)
      + sum_c [
            log Gamma(1/3 + N_m[c])
          - log Gamma(1/3)
        ]

    log_L_bias = -2.1986

Model posterior:

    p_resp = 0.0038
    p_bias = 0.9962

Bias posterior mean:

    E[r_m] =
        ([1/3, 1/3, 1/3] + [6, 0, 0]) / 7

    E[r_m] =
        [0.9048, 0.0476, 0.0476]

For a candidate with classifier probabilities:

    P[n] = [0.8, 0.1, 0.1]

the bias branch score is:

    bias_score =
        0.8 * 0.9048
      + 0.1 * 0.0476
      + 0.1 * 0.0476

    bias_score = 0.7333

This is not credited as local responsive skill. It is credited as a likely
response-bias match.


Example 5: Gated std-UCB
~~~~~~~~~~~~~~~~~~~~~~~~
Suppose:

    mean_G = 0.70
    std_G = 0.10
    mean_L = 0.50
    std_L = 0.20
    lambda = 0.25
    global_exploration_weight = 1.0
    local_exploration_weight = 1.0
    constraint_pressure = 0.6
    constraint_damps_global_exploration = true

Then:

    w_G = 1.0 * (1 - 0.6) = 0.4
    w_L = 1.0

    ucb_G = 0.70 + 0.4 * 0.10 = 0.74
    ucb_L = 0.50 + 1.0 * 0.20 = 0.70

    responsive_ucb =
        0.75 * 0.74 + 0.25 * 0.70 = 0.73

The constraint dampens global annotator exploration, but local uncertainty still
receives the full local exploration bonus.


Example 6: Gated randomized std-UCB
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Suppose the same values as Example 5, but:

    score_mode = "randomized_ucb"
    z_G[m] = 0.2
    z_L[n, m] = 0.75

Then:

    w_G = 1.0 * (1 - 0.6) * 0.2 = 0.08
    w_L = 1.0 * 0.75 = 0.75

    ucb_G = 0.70 + 0.08 * 0.10 = 0.708
    ucb_L = 0.50 + 0.75 * 0.20 = 0.65

    responsive_ucb =
        0.75 * 0.708 + 0.25 * 0.65 = 0.6935

The global multiplier is sampled per annotator and damped by assignment
pressure. The local multiplier is sampled per pair and is not damped by
assignment pressure.


Recommended ablations
---------------------
Study 1 selected BLGA candidate:

    locality_mode=local
    local_evidence_mode=kernel
    responsive_combination=gated
    bias_model_correction=model_average
    evidence_weighting=confidence
    agreement_mode=soft_chance_corrected
    bias_response_weighting=uniform
    score_mode=randomized_ucb
    ucb_mode=std
    random_ucb_values=[0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    random_ucb_probs=[0.35, 0.30, 0.18, 0.09, 0.05, 0.02, 0.01]
    constraint_damps_global_exploration=true
    gate_constraint_coupling=linear
    use_rho_correction=false

Policy ablations, with all other selected BLGA settings fixed:

    score_mode=mean
    score_mode=thompson, thompson_samples=5
    score_mode=ucb, ucb_mode=std
    score_mode=randomized_ucb, ucb_mode=std

Global/local structure ablation:

    locality_mode=global
    locality_mode=local

Agreement evidence ablations:

    agreement_mode=argmax
    agreement_mode=soft_chance_corrected
    agreement_mode=soft_raw_probability

Evidence mass ablations:

    evidence_weighting=confidence
    evidence_weighting=entropy
    evidence_weighting=margin
    evidence_weighting=uniform

Response-bias and gate ablations:

    bias_model_correction=model_average
    bias_model_correction=none
    gate_constraint_coupling=linear
    gate_constraint_coupling=none

Fixed for the main Study 1 grid unless explicitly listed above:

    local_evidence_mode=kernel
    use_rho_correction=false

Study 1 launch grids use `optimal_sample_loose_cap` and cross all BLGA variants
with:

    max_per_annotator_multiplier=1.0
    max_per_annotator_multiplier=2.0
    max_per_annotator_multiplier=4.0


Evaluation studies
------------------
Study 1 is the BLGA development and selection study. It uses DTD47 variants as
the held-out design test bed, random sample selection, `annot_mix_gating` and
`geo_reg_f`, and all three cap multipliers. The primary endpoint is
budget-weighted AUC of `delta_new_pair_acc`, excluding the shared initial cycle.
After Study 1, the BLGA configuration is frozen.

Study 2 is the main annotator-selection benchmark. It uses random sample
selection and compares frozen `blga_gated_randomized_ucb` against:

    random
    performance
    beta
    label_minority
    representation_diversity
    semantic_diversity
    agreement_global
    label_quality_global
    crowdlab_global
    ig

The Study 2 datasets are all non-DTD47 datasets with annotator labels available
through either simulation or real `z` labels:

    al_rcta_agnews
    al_rcta_consumer_complaints
    al_rcta_wiki_movie_plots
    audiomnist10
    banking77
    dermamnist7
    dopanim
    food101
    letter26
    skits2i14
    trec6

Study 2 crosses both crowd classifiers and the three cap multipliers. The
primary endpoint is budget-weighted AUC of `delta_new_pair_acc`, excluding the
initial cycle. `test_acc`, `acc_pair_micro`, and `acc_mv` are secondary.
Results should be summarized by pairwise win rates and average ranks, with
stratified views by annotator source, classifier, and cap multiplier.

Study 3 is a smaller active-learning interaction study. It keeps BLGA frozen
and uses the current instance-first, annotator-second protocol: `sample.actual`
first selects candidate samples, then the annotator scorer and
`optimal_sample_loose_cap` assign annotators to those candidates. The main
samplers are:

    random
    margin
    badge

The main annotator scorers are:

    random
    performance
    blga_gated_randomized_ucb

The initial Study 3 dataset subset is:

    banking77
    dermamnist7
    skits2i14
    al_rcta_wiki_movie_plots

Study 3 still crosses both crowd classifiers and all three cap multipliers. The
primary endpoint is AUC of `test_acc` over cycles, with acquired-label-quality
AUC as a secondary diagnostic. Results should emphasize sampler-by-scorer
interactions and include learning curves only for representative cases.


Known limitations and open questions
------------------------------------
- Agreement is classifier-conditioned. Confident classifier errors can misrank
  annotators.
- `agreement_mode="soft_raw_probability"` is closest to classifier label
  quality, but can treat labels far above random chance as mostly failure in
  large-class problems.
- `agreement_mode="soft_chance_corrected"` avoids this large-class issue, but
  labels at or below random classifier probability are clipped to zero support.
- `rho` is a broadness penalty, not a proof of useful locality.
- Gated `lambda` is a heuristic model-average gate, not a fully Bayesian
  posterior. With `gate_constraint_coupling="none"`, it is interpreted as
  epistemic trust in local classifier-conditioned evidence. With
  `gate_constraint_coupling="linear"`, it also encodes assignment pressure.
- Deterministic UCB is interpretable and reproducible, but it does not fully
  solve lock-in under weak constraints. Randomized UCB addresses this by
  randomizing optimism, while Thompson sampling remains an important posterior
  randomization ablation.
- Batch selection uses a static utility matrix. Hypothetical posterior updates
  within a batch are not currently encoded in the exact assigner objective.
- `k_star` remains a square-root budget heuristic. The scorer splits its roles
  through multipliers, and `projected_annotator_budget` can make it
  annotator-specific, but this is still an approximation rather than an exact
  decision-theoretic neighborhood rule.
- Kernel mode currently supports only the `full_kth` bandwidth rule.
- Classwise balanced local agreement remains an optional diagnostic for small
  class counts, not the main scorer.
- Toy-widget alignment is still a separate follow-up.
