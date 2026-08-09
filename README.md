We found a paper that was using participation ratio (PR) and 2 nearest neighbours (2NN) to estimate the 'dimensionality' of pictures, and then comparing them for natural images and those from the manifolds of different models with varing robustness.

<img width="2638" height="724" alt="image" src="https://github.com/user-attachments/assets/da46c0fb-e19e-4a95-bf29-53eb88076d5f" />

We then had a few critiques, which are summarized here in an interactive marimo notebook: https://molab.marimo.io/notebooks/nb_2y5KGCjDu3UzNEYQhcdbnf

# Exponential Extension

Critique and replication notes for  
[Solving adversarial examples requires solving exponential misalignment](https://arxiv.org/abs/2603.03507).

This repo explores whether PR and 2NN “dimensionality” estimates used for perceptual manifolds (PMs) are robust, or whether they can be inflated/deflated by simple, semantically irrelevant image manipulations.

## Main claim we test

If PR/2NN are treated as estimates of a clean semantic class manifold, they should be stable under small, human-irrelevant transforms.  
In practice, they are highly sensitive to high-frequency / nuisance pixel structure.

## What we ran

### 1. Manifold (PM) sampling
We sampled high-confidence “cat” images via projected gradient ascent on RobustBench CIFAR-10 L∞ models:

- `Standard`
- `Wong2020Fast`
- `Wu2020Adversarial_extra`
- `Peng2023Robust`

Script: `gradient_diffusion_experiments.py`  
Example outputs: `outputs/gradient_diffusion_experiments/`

### 2. Manipulation stress tests (PR / 2NN)
We applied all combinations of:

- random noise
- low-pass Gaussian
- high-pass Gaussian
- rainbow border

to:

- natural STL10 cats
- synthetic Peng2023Robust cats
- pure random noise

Script: `cat_manipulation_pr2nn.py`  
UI notebook: `cat_manipulation_marimo_ui.ipynb`

**Primary metrics run (CSV only):**  
`outputs/cat_manipulation_pr2nn/20260426_214917/metrics.csv`

Setup:
- feature side `24` → ambient dim `D = 1728`
- natural cats / noise: **N = 3000**
- synthetic Peng cats: **N = 20** (limited available PM samples)

## Key findings

### 1. Random noise is high-dimensional once N is large enough
With N=3000, pure noise reaches very high PR (~927), near the ambient-feature regime.  
Earlier N=20 runs looked “low PR” only because PR is capped near `N-1`.

| Dataset | Combo | N | PR | 2NN |
|---|---|---:|---:|---:|
| natural_cat | none | 3000 | ~12.0 | ~1.1 |
| random_noise | none | 3000 | ~926.9 | ~212.1 |
| synthetic_cat_peng2023robust | none | 20 | ~16.4 | ~42.9 |

### 2. High-pass filtering strongly inflates natural-image PR
Natural cats jump from PR ≈ 12 to PR ≈ 158 under high-pass alone; noise+high-pass goes even higher (~181).

Interpretation: PR responds strongly to high-frequency pixel variation, not just semantic “cat-ness.”

### 3. Tiny nuisance edits can move 2NN a lot
Adding random noise to natural cats barely changes PR (~12.0 → ~12.2), but 2NN jumps dramatically (~1.1 → ~18.9).

Interpretation: 2NN is especially sensitive to local pixel-level degrees of freedom.

### 4. Rainbow borders / low-pass are milder but still shift estimates
- Low-pass slightly lowers natural PR (~12.0 → ~10.5)
- Rainbow border mildly lowers natural PR (~12.0 → ~11.6)
- Combined high-pass + border can push natural PR above 180

### 5. Synthetic PM samples are not yet fair apples-to-apples
Peng synthetic metrics remain N=20 constrained, so PR stays near the sample-size ceiling and cannot be compared directly to N=3000 natural/noise numbers.

## Conceptual takeaway

PR and 2NN are easy to “hack” as semantic-manifold dimensionality measures:

- inflate natural images with high-frequency / nuisance structure
- deflate / reshape PM-looking sets by changing initialization / filtering

So high PM dimension vs natural-image dimension can reflect estimator sensitivity and sampling procedure, not necessarily a clean semantic geometry result.

## Repo map

- `nn_pr_visual_explainer_marimo.ipynb` — intuition for 2NN + PR
- `paper_pr_replica_marimo.ipynb` — PR across STL10 natural images
- `pm_sampling_paper_style_minimal_marimo.ipynb` — simplified PM sampling demo
- `gradient_diffusion_experiments.py` — PM sampling across RobustBench models
- `cat_manipulation_pr2nn.py` — manipulation grid + metrics CSV
- `cat_manipulation_marimo_ui.ipynb` — interactive checklist UI over metrics

## Reproduce (metrics only, no image dumps)

```bash
python cat_manipulation_pr2nn.py \
  --num-images 3000 \
  --allow-replacement \
  --no-save-images

## Caveats
- Synthetic Peng set size is currently small (N=20).
- Feature resolution (feature_side=24) affects absolute PR/2NN values.
- Replacement sampling was used to reach N=3000 STL10 cats.
- These experiments critique interpretation of the estimators; they do not by themselves fully refute the paper’s PM definition.
