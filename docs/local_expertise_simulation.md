# Local-Expertise Simulation

The multi-annotator simulator uses a GLAD/IRT-style ability-difficulty model
with an optional feature-local expertise term. The goal is to stress-test
annotator-selection methods under controlled assumption mismatch: global,
mixed, and local annotator reliability.

## Model

For a normal annotator `a` and sample `i`, corrected skill is modeled on a
latent logistic scale:

```text
eta_ai = logit(q_a) - beta_a * difficulty_i + local_effect_ai
q_ai = sigmoid(eta_ai)
p_correct_ai = chance + (1 - chance) * q_ai
```

Here, `chance = 1 / n_classes` and `q` is chance-corrected accuracy:

```text
q = (accuracy - chance) / (1 - chance)
```

If the sampled annotation is incorrect, the wrong label is drawn uniformly from
the incorrect classes. Uniform and single-class spammers ignore ability,
difficulty, and local expertise.

## Local Expertise

Feature-local regimes give each normal annotator three class-balanced prototype
samples. Its local score is the maximum RBF similarity to these prototypes. The
bandwidth is set by a dataset distance quantile, so the neighborhood size is
interpretable across dataset sizes.

Class-dependent regimes avoid embedding similarity entirely. Each normal
annotator receives one or more specialist true classes, and its local score is
one for samples from those classes and zero otherwise. This is useful as a
sanity-check regime when pretrained embeddings do not provide a meaningful
similarity notion.

The local score is centered per annotator and scaled so the expected
top-quartile versus bottom-quartile local-effect gap approximately matches:

```text
local_expertise_target_gap_q * local_variability
```

Default archetype variability:

```text
expert:    0.6
competent: 1.0
novice:    1.3
spammers:  ignored
```

## Regimes

```text
global: local_expertise_enabled = false, target gap = 0.00
mixed:  local_expertise_enabled = true,  target gap = 0.15
local:  local_expertise_enabled = true,  target gap = 0.30
class_extreme: class-dependent target gap = 0.55
```

Use Hydra overrides such as:

```bash
python scripts/experiment.py dataset=trec6 al=trec6 simulation=trec6 simulation/regime=local
```

## Diagnostics

Diagnostics are printed only on cache miss. They report the target, expected
local-effect gap, total expected gap, sampled gap, and per-archetype aggregates.
These numeric diagnostics are the simulator validation source of truth; t-SNE
plots are only qualitative illustrations.

## Motivation

The simulator follows the established ability-difficulty view from GLAD/IRT
models and adds input-space-dependent annotator expertise. This keeps the
simulation explainable while allowing controlled global-to-local stress tests.
