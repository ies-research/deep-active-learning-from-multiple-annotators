# KernelSmoothedBayesianAnnotatorGainNew Presets

Preset names follow:

```text
<evidence_scope>_<posterior_family>_<prior_source>_prior
```

`evidence_scope` is `global` or `local`. Global evidence pools all observed labels.
Local evidence uses kernel weights around each candidate instance.

`posterior_family` is one of:

- `full`: full confusion Dirichlet posterior, `P(label=c | true=z, annotator=m)`.
- `mace`: MACE-style accuracy plus class-independent response bias.
- `accuracy_uniform`: one accuracy per annotator; wrong-label mass is uniform.

`prior_source` is one of:

- `fixed`: identical annotator-independent fixed confusion prior.
- `global_full`: empirical global full-confusion prior.
- `global_mace`: empirical global MACE prior.
- `global_accuracy_uniform`: empirical global accuracy-uniform prior.
- `local_mace`: local empirical MACE prior, only used to regularize local full posteriors.
- `local_accuracy_uniform`: local empirical accuracy-uniform prior, only used to regularize local full posteriors.

## Fixed Priors

Fixed prior means are controlled by one scalar:

```yaml
fixed_prior_accuracy: 0.99
prior_strength: 20.0
```

`fixed_prior_accuracy` is the actual diagonal probability of the fixed prior channel.
If it is omitted or `null`, it defaults to chance level, `1 / K`.

For `K` classes, the fixed prior channel is:

```text
B[z, c] =
  fixed_prior_accuracy                 if c == z
  (1 - fixed_prior_accuracy) / (K - 1) if c != z
```

`prior_strength` controls how strongly the resulting base prior contributes to the
final posterior. For full Dirichlet posteriors, it is the total pseudo-count mass per
row. For MACE and accuracy-uniform posteriors, it is the pseudo-count strength of the
restricted response parameters.

For empirical-Bayes prior sources, add:

```yaml
fixed_prior_strength: 1.0
```

`fixed_prior_strength` controls how strongly `fixed_prior_accuracy` is used while
estimating empirical base priors, for example `*_global_mace_prior` or
`*_global_full_prior`.

## Canonical Presets

Full-confusion posterior:

```yaml
global_full_fixed_prior
global_full_global_full_prior
global_full_global_mace_prior
global_full_global_accuracy_uniform_prior

local_full_fixed_prior
local_full_global_full_prior
local_full_global_mace_prior
local_full_global_accuracy_uniform_prior
local_full_local_mace_prior
local_full_local_accuracy_uniform_prior
```

MACE posterior:

```yaml
global_mace_fixed_prior
global_mace_global_mace_prior
global_mace_global_full_prior

local_mace_fixed_prior
local_mace_global_mace_prior
local_mace_global_full_prior
```

Accuracy-uniform posterior:

```yaml
global_accuracy_uniform_fixed_prior
global_accuracy_uniform_global_accuracy_uniform_prior
global_accuracy_uniform_global_full_prior

local_accuracy_uniform_fixed_prior
local_accuracy_uniform_global_accuracy_uniform_prior
local_accuracy_uniform_global_full_prior
```

Local balanced-accuracy posterior:

```yaml
local_balanced_accuracy_global_full_prior
```

## Examples

Optimistic local full-confusion posterior with no empirical base estimate:

```yaml
preset: local_full_fixed_prior
fixed_prior_accuracy: 0.99
prior_strength: 20.0
```

Local full-confusion posterior regularized by an empirical global MACE prior:

```yaml
preset: local_full_global_mace_prior
fixed_prior_accuracy: null
fixed_prior_strength: 1.0
prior_strength: 5.0
```

Local MACE posterior with an optimistic fixed prior:

```yaml
preset: local_mace_fixed_prior
fixed_prior_accuracy: 0.95
prior_strength: 10.0
```

Local accuracy-uniform posterior with an empirical global full-confusion prior:

```yaml
preset: local_accuracy_uniform_global_full_prior
fixed_prior_accuracy: 0.9
fixed_prior_strength: 5.0
prior_strength: 10.0
```

## Parameter Notes

`fixed_prior_accuracy` and `fixed_prior_strength` are used by all fixed and empirical
prior sources, independent of the posterior family.

## Budget-Aware Locality

Local presets can choose their local support size from the annotation budget instead
of using a fixed neighborhood size:

```yaml
budget_aware_locality: true
budget_T0: 10.0
budget_rho: 1.0
budget_k_min: 20
budget_k_max: 500
budget_s_min: 5.0
budget_s_max: 50.0
local_kernel_weighting: kernel
```

At scoring time, `scripts/experiment.py` passes the total pair budget to the scorer.
The scorer infers the used budget from the currently observed labels in `y`.

The rule is:

```text
k_final = ceil(budget_T0 * N * M / B_total)
k_t     = ceil(budget_T0 * N * M / B_t)
k_t     = clip(k_t, max(k_final, budget_k_min), budget_k_max)
s_local = clip(budget_rho * k_t * B_t / (N * M), budget_s_min, budget_s_max)
```

`k_t` is an actual top-k support cap for local evidence. Kernel weights are not
renormalized after truncation. The implementation also caps `k_t` at `N`, because a
neighborhood cannot contain more pool instances than exist.

`local_kernel_weighting` controls weights inside the retained neighborhood:

- `kernel`: keep the original RBF/cosine kernel weights.
- `constant`: use the kernel only to select the top-k support, then assign weight
  `1.0` to every retained neighbor.

`s_local` overrides `prior_strength` only for local posterior combinations and local
balanced-accuracy estimates. Global posteriors continue to use `prior_strength`.

During experiments, enabled budget-aware locality prints one line per cycle:

```text
[budget_aware_locality] cycle=... k_t=... k_final=... s_local=...
```

Diagnostics are available after scoring:

```python
scorer.last_budget_aware_locality_
scorer.last_local_kernel_top_k_
scorer.last_local_prior_strength_
```

If `last_budget_aware_locality_.diagnostics["k_final_over_N"] > 0.2`, the total
budget is probably too small for fine local annotator-specific reliability estimates.
